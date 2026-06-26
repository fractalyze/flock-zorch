"""SHA-256 Merkle-tree byte-match gate.

Loads flock's golden (`merkle_root` over N leaves) and asserts the jax port
reproduces the 32-byte root bit-for-bit — the oracle gate. Merkle is a <1% PCS
component (NTT dominates 96-322x, see README), and CPU SHA-NI wins the hashing,
so the gate here is correctness, not a 10x speed target; GPU time is informational.

Run:
  cargo run --release --example dump_merkle -- 4096 64 artifacts/merkle_golden.bin
  JAX_PLATFORMS=cuda PYTHONPATH=python <venv> python/flock_zorch/testing/merkle_oracle_test.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax

from flock_zorch import merkle

ART = Path(__file__).resolve().parents[3] / "artifacts"


def _load_golden():
    raw = (ART / "merkle_golden.bin").read_bytes()
    assert raw[:8] == b"FLKMRK01", "bad magic"
    n_leaves = int.from_bytes(raw[8:16], "little")
    leaf_size = int.from_bytes(raw[16:24], "little")
    off = 24
    data = np.frombuffer(raw, np.uint8, n_leaves * leaf_size, off).reshape(n_leaves, leaf_size)
    root = np.frombuffer(raw, np.uint8, 32, off + n_leaves * leaf_size)
    return n_leaves, leaf_size, data, root


def main() -> int:
    n_leaves, leaf_size, data, golden_root = _load_golden()
    print(f"device: {jax.devices()[0]} | backend: {jax.default_backend()}")
    got = merkle.merkle_root(data)
    ok = np.array_equal(got, golden_root)
    print(f"Merkle root byte-identity vs flock ({n_leaves} x {leaf_size}B leaves): "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        print(" got :", bytes(got).hex())
        print(" want:", bytes(golden_root).hex())
        return 1

    # Informational GPU timing (best-of-N).
    t0 = time.perf_counter()
    for _ in range(20):
        merkle.merkle_root(data)
    print(f"GPU merkle_root ({n_leaves} leaves): {(time.perf_counter()-t0)/20*1e3:.3f} ms "
          f"(informational; Merkle is <1% of PCS commit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
