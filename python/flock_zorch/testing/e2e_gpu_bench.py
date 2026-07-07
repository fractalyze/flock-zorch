"""End-to-end GPU prover timing vs flock's CPU prove_fast.

Times each byte-identical GPU layer (pcs.commit + zerocheck + lincheck + pcs.open)
at the prover's sizes with representative inputs, keeping the codeword/tree
device-resident, and reports the per-phase split + total vs flock's measured CPU
prove_fast. (Per-layer byte-identity is proven by the *_oracle_test gates; this is
the speed assembly.) Params mirror the blake3 prover: log_inv_rate=1,
log_batch_size=5.

Run:
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH=python:/home/jooman/fractalyze/zorch <venv> \
      python/flock_zorch/testing/e2e_gpu_bench.py [m ...]
"""
import os
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jax import lax  # noqa: E402
from zorch.coding.additive_reed_solomon import AdditiveReedSolomon  # noqa: E402

from flock_zorch import (field, pcs_commit, zerocheck,  # noqa: E402
                         lincheck, pcs_open, merkle)
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.testing._util import best  # noqa: E402

LIR, LBS = 1, 5
# Route Merkle (commit root + BaseFold T2/epoch trees) through flock's host SHA-NI
# instead of GPU SHA-256 — the "SHA-off-GPU" optimization (byte-identical; gated).
HOST_SHA = os.environ.get("FLOCK_HOST_SHA") == "1"
# flock CPU prove_fast (blake3_proof bench, this box): m -> ms.
CPU_PROVE_FAST = {20: 19.94 * 4, 22: 19.94, 26: 218.73, 28: 940.01}  # m=20 approx scale


def bench(m):
    rng = np.random.default_rng(7)
    log_msg = m - 7
    n_packed = 1 << log_msg
    z_packed = jnp.asarray(rng.integers(0, 2**64, size=(n_packed, 2), dtype=np.uint64))
    k_code = (log_msg - LBS) + LIR
    num_ntts = 1 << LBS

    # --- commit (RS encode + merkle) ---
    n_pos_msg, n_pos_code = 1 << (log_msg - LBS), 1 << k_code
    code = AdditiveReedSolomon(n_pos_msg, 1 << LIR, jnp.binary_field_ghash)
    enc = jax.jit(lambda z: code.encode(lax.bitcast_convert_type(
        z.reshape(n_pos_msg, num_ntts, 2), jnp.binary_field_ghash).T))  # [num_ntts, n_pos_code]
    cw = enc(z_packed); cw.block_until_ready()
    t_commit = best(lambda: enc(z_packed), n=4)
    cw_np = np.frombuffer(
        np.ascontiguousarray(np.asarray(cw).T).tobytes(), np.uint64).reshape(n_pos_code * num_ntts, 2)
    leaves = cw_np.reshape(n_pos_code, num_ntts * 2).view(np.uint8)
    t_merkle = best(lambda: merkle.merkle_tree(leaves, use_host_sha=HOST_SHA), n=2)
    init_tree = merkle.merkle_tree(leaves, use_host_sha=HOST_SHA)

    # --- zerocheck (random a/b/c bits) ---
    a = rng.integers(0, 2, size=1 << m, dtype=np.uint8)
    b = rng.integers(0, 2, size=1 << m, dtype=np.uint8)
    c = rng.integers(0, 2, size=1 << m, dtype=np.uint8)
    t_zc = best(lambda: zerocheck.prove_packed(a, b, c, m, b"e2e"), n=3)

    # --- lincheck (k_log/k_skip like flock; sparse A0/B0) ---
    k_log, k_skip = 7, 6
    k = 1 << k_log
    A = (rng.integers(0, 2, size=(k, k)) & (rng.integers(0, 4, size=(k, k)) == 0)).astype(np.uint64)
    B = (rng.integers(0, 2, size=(k, k)) & (rng.integers(0, 4, size=(k, k)) == 0)).astype(np.uint64)
    n_log = m - k_log
    zp_lin = rng.integers(0, 256, size=((1 << n_log) // 8) * k, dtype=np.uint8).tobytes()
    x_ab = {"z_skip": rng.integers(0, 2**64, size=2, dtype=np.uint64),
            "x_inner_rest": rng.integers(0, 2**64, size=(k_log - k_skip, 2), dtype=np.uint64),
            "x_outer": rng.integers(0, 2**64, size=(n_log, 2), dtype=np.uint64)}
    t_lc = best(lambda: lincheck.prove(zp_lin, A, B, x_ab, m, k_log, k_skip), n=3)

    # --- pcs.open ---
    x_outer = jnp.asarray(rng.integers(0, 2**64, size=(m - 6, 2), dtype=np.uint64))
    def open_fn():
        ch = Challenger(b"e2e")
        return pcs_open.open(z_packed, codeword, init_tree, x_outer, k_code, LIR, LBS, ch,
                             use_host_sha=HOST_SHA)
    t_open = best(open_fn, n=2)

    gpu = t_commit + t_merkle + t_zc + t_lc + t_open
    cpu = CPU_PROVE_FAST.get(m)
    print(f"\nm={m} (k_code={k_code}, params rate=1/2 batch=2^{LBS}):")
    print(f"  commit(NTT) {t_commit:7.2f}  merkle {t_merkle:7.2f}  zerocheck {t_zc:7.2f}  "
          f"lincheck {t_lc:7.2f}  open {t_open:7.2f}  ms")
    print(f"  GPU total {gpu:8.2f} ms   |   CPU prove_fast {cpu:8.2f} ms   =>  {cpu/gpu:.1f}x")
    return cpu / gpu


def main():
    ms = [int(x) for x in sys.argv[1:]] or [22, 26]
    print(f"device: {jax.devices()[0]} | mul: software"
          f" | Merkle: {'HOST SHA-NI' if HOST_SHA else 'GPU SHA-256'}")
    for m in ms:
        bench(m)


if __name__ == "__main__":
    main()
