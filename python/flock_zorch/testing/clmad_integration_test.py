"""clmad integration gate + benchmark.

Validates that the FFI clmad path (`field_clmad.mul`) is byte-identical to
flock's GF(2^128) multiply golden, then measures the bare-multiply speedup over
the software `field.mul`. clmad is the GPU field multiply the BaseFold open's
SoA steps (row-batch collapse, round message, a/b fold) can run on. Needs the
built FFI handler (optim/clmad/build_ffi.sh) + an sm_120 GPU.
"""
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flock_zorch import field, field_clmad  # noqa: E402

ART = Path(__file__).resolve().parents[3] / "artifacts"


def _load_field_golden():
    raw = (ART / "field_mul_golden.bin").read_bytes()
    n = int.from_bytes(raw[8:16], "little")
    off, blk = 16, n * 16
    g = lambda o: np.frombuffer(raw, np.uint64, n * 2, o).reshape(n, 2)  # noqa: E731
    return n, g(off), g(off + blk), g(off + 2 * blk)


def test_field_clmad_oracle():
    n, a, b, golden = _load_field_golden()
    got = np.asarray(jax.jit(field_clmad.mul)(jnp.asarray(a), jnp.asarray(b)))
    assert np.array_equal(got, golden), "field_clmad.mul != flock golden"


def main():
    if not field_clmad.available():
        print("clmad FFI not built — run optim/clmad/build_ffi.sh")
        return
    test_field_clmad_oracle()
    print("field_clmad == flock golden: PASS")

    print("\nGF(2^128) multiply: software vs clmad")
    for log in (16, 18, 20):
        n = 1 << log
        a = jnp.asarray(np.random.default_rng(3).integers(0, 2**64, size=(n, 2), dtype=np.uint64))
        b = jnp.asarray(np.random.default_rng(4).integers(0, 2**64, size=(n, 2), dtype=np.uint64))
        row = f"  n=2^{log:<2}"
        for name, mulf in (("software", field.mul), ("clmad", field_clmad.mul)):
            fn = jax.jit(mulf)
            r = fn(a, b)
            r.block_until_ready()
            it = 30
            t0 = time.perf_counter()
            for _ in range(it):
                r = fn(a, b)
            r.block_until_ready()
            row += f"  {name}={(time.perf_counter()-t0)/it*1e3:7.2f}ms"
        print(row)


if __name__ == "__main__":
    main()
