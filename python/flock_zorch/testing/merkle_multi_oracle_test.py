"""Merkle tree + octopus multi-proof byte-match gate vs flock.

Loads flock's golden (tree data + query positions + multi-proof) and asserts the
jax port reproduces both the tree (implicitly, via the proof hashes) and the
multi-proof byte-for-byte — the query-opening primitive of the PCS open.

Run:
  cargo run --release --example dump_merkle_multi -- 4096 64 30 artifacts/merkle_multi_golden.bin
  JAX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/merkle_multi_oracle_test.py
"""
import sys
from pathlib import Path

import numpy as np
import jax

from flock_zorch import merkle

ART = Path(__file__).resolve().parents[3] / "artifacts"


def main() -> int:
    raw = (ART / "merkle_multi_golden.bin").read_bytes()
    assert raw[:8] == b"FLKMMP01", "bad magic"
    n_leaves = int.from_bytes(raw[8:16], "little")
    leaf_size = int.from_bytes(raw[16:24], "little")
    n_pos = int.from_bytes(raw[24:32], "little")
    off = 32
    data = np.frombuffer(raw, np.uint8, n_leaves * leaf_size, off).reshape(n_leaves, leaf_size)
    off += n_leaves * leaf_size
    positions = np.frombuffer(raw, np.uint64, n_pos, off).astype(np.int64); off += n_pos * 8
    proof_len = int.from_bytes(raw[off:off + 8], "little"); off += 8
    golden = np.frombuffer(raw, np.uint8, proof_len * 32, off).reshape(proof_len, 32)

    print(f"device: {jax.devices()[0]} | backend: {jax.default_backend()}")
    tree = merkle.merkle_tree(data)
    got = merkle.merkle_multi_proof(tree, n_leaves, positions)
    ok = got.shape == golden.shape and np.array_equal(got, golden)
    print(f"merkle_multi_proof byte-match vs flock ({n_leaves} leaves, {n_pos} positions, "
          f"proof_len={proof_len}): {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(" got shape", got.shape, "want", golden.shape)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
