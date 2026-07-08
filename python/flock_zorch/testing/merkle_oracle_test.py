"""SHA-256 Merkle-tree byte-match gate.

Loads flock's golden (`merkle_root` over N leaves) and asserts the jax port
reproduces the 32-byte root bit-for-bit — the oracle gate. Merkle is a <1% PCS
component (NTT dominates 96-322x, see README); the gate here is correctness, not a
speed target; GPU time is informational.

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


def _hooks_on_commit_path() -> bool:
    """Pin the subclass seam: zorch's commit must route through the two batching
    hooks. They are underscore-private upstream, so a pin bump that renames or
    bypasses them would fall back to the byte-identical vmap-of-single path —
    every byte gate stays green while the batch-native `zorch.sha256` marker
    (the reason `_Sha256Merkle` exists) silently evaporates."""
    calls = set()
    cls = merkle._Sha256Merkle
    orig_h, orig_c = cls._hash_leaves, cls._compress_groups
    cls._hash_leaves = lambda self, m: calls.add("leaves") or orig_h(self, m)
    cls._compress_groups = lambda self, g: calls.add("compress") or orig_c(self, g)
    try:
        merkle.merkle_root(np.zeros((4, 64), np.uint8))
    finally:
        cls._hash_leaves, cls._compress_groups = orig_h, orig_c
    return calls == {"leaves", "compress"}


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

    # disable_jit so `_root_dev` runs eagerly: the probe observes the Python
    # routing hooks, which a persistent-compilation-cache hit would bypass.
    with jax.disable_jit():
        hooks_ok = _hooks_on_commit_path()
    print(f"batch hooks on zorch commit path: {'PASS' if hooks_ok else 'FAIL'}")
    if not hooks_ok:
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
