"""GPU per-phase prover timing, to pair with flock's CPU `prove_fast` breakdown
(blake3_proof bench) for an end-to-end GPU-vs-CPU comparison.

Times the GPU compute of the dominant field-arithmetic phases at a given m, with
clmad: pcs::commit (pack+interleaved NTT) and zerocheck::prove_packed (URM +
multilinear sumcheck). lincheck is tiny; pcs::open (FRI folds) is NTT-character
(projected from commit until ported). All on the zorch venv.

Run:
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/prover_phase_gpu_bench.py [m ...]
"""
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flock_zorch import zerocheck  # noqa: E402
from flock_zorch.pcs import commit as pcs_commit  # noqa: E402

from flock_zorch.testing._util import best  # noqa: E402

LOG_INV_RATE = 1
LOG_BATCH = 5


def bench_commit(m):
    n_packed = 1 << (m - 7)
    z_packed = jnp.asarray(np.random.default_rng(1).integers(0, 2**64, size=(n_packed, 2), dtype=np.uint64))
    return best(lambda: pcs_commit.commit_root(z_packed, m, LOG_INV_RATE, LOG_BATCH), 5)


def bench_zerocheck(m):
    # Random bits (timing is value-independent); prove_packed runs URM + sumcheck.
    rng = np.random.default_rng(2)
    a = rng.integers(0, 2, size=1 << m, dtype=np.uint8)
    b = rng.integers(0, 2, size=1 << m, dtype=np.uint8)
    c = rng.integers(0, 2, size=1 << m, dtype=np.uint8)
    # prove_packed is a host loop (jit'd kernels inside); time the whole call.
    return best(lambda: zerocheck.prove_packed(a, b, c, m, b"flock-bench-v0"), n=3)


def main():
    ms = [int(x) for x in sys.argv[1:]] or [20, 22, 24]
    print(f"device: {jax.devices()[0]} | params: rate=1/2^{LOG_INV_RATE} batch=2^{LOG_BATCH}")
    print(f"{'m':>3}  {'commit(GPU)':>12}  {'zerocheck(GPU)':>15}")
    for m in ms:
        c = bench_commit(m)
        z = bench_zerocheck(m)
        print(f"{m:>3}  {c:>10.3f}ms  {z:>13.3f}ms")


if __name__ == "__main__":
    main()
