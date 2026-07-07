"""Unit test for the octopus↔per-query-path pair in `merkle.py`:
`multi_proof_to_paths` (octopus→paths, the expander that lets the BaseFold
verifier feed flock's octopus multi-proof into zorch's `pcs.fold.verify_openings`)
and its inverse `paths_to_multi_proof` (paths→octopus, the prover-side assembler
that rebuilds flock's octopus from a zorch `Opening`'s per-query paths).

Pure host (hashlib reference tree + flock's own `merkle_multi_proof`), no GPU:
these are proof-format decoding, not field math. Verifies each reconstructed
path equals the tree's actual siblings AND rebuilds the root, then round-trips
paths→octopus back to the original multi-proof byte-for-byte.
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np

from flock_zorch import merkle


def _ref_tree(leaves: np.ndarray) -> np.ndarray:
    """flock's flat Merkle tree via real SHA-256 (leaf = SHA256(bytes), node =
    SHA256(l‖r)); layout tree[0..n]=leaf hashes, then each level up, root last."""
    n = leaves.shape[0]
    nodes = [hashlib.sha256(leaves[i].tobytes()).digest() for i in range(n)]
    tree = list(nodes)
    level = nodes
    while len(level) > 1:
        nxt = [
            hashlib.sha256(level[i] + level[i + 1]).digest()
            for i in range(0, len(level), 2)
        ]
        tree.extend(nxt)
        level = nxt
    return np.stack([np.frombuffer(h, np.uint8) for h in tree])


def _level_starts(num_leaves: int) -> list[int]:
    starts, s, ln = [], 0, num_leaves
    while ln >= 1:
        starts.append(s)
        s += ln
        if ln == 1:
            break
        ln >>= 1
    return starts


def _check(num_leaves: int, leaf_bytes_len: int, positions: list[int], name: str) -> bool:
    rng = np.random.default_rng(len(positions) + num_leaves)
    leaves = rng.integers(0, 256, size=(num_leaves, leaf_bytes_len), dtype=np.uint8)
    tree = _ref_tree(leaves)
    root = tree[-1]
    starts = _level_starts(num_leaves)
    depth = num_leaves.bit_length() - 1

    proof = merkle.merkle_multi_proof(tree, num_leaves, positions)
    leaf_bytes = np.stack([leaves[p] for p in positions])  # [Q, L]
    paths = merkle.multi_proof_to_paths(proof, num_leaves, positions, leaf_bytes)

    ok = paths.shape == (len(positions), depth, 32)
    # Each query's path[k] must equal the tree's sibling of (p>>k).
    for qi, p in enumerate(positions):
        for k in range(depth):
            node = p >> k
            expected = tree[starts[k] + (node ^ 1)]
            if not np.array_equal(paths[qi, k], expected):
                ok = False
        # Rebuild the root from leaf hash + path (parity-ordered compress).
        node_hash = hashlib.sha256(leaves[p].tobytes()).digest()
        idx = p
        for k in range(depth):
            sib = paths[qi, k].tobytes()
            cur = node_hash
            node_hash = (
                hashlib.sha256(cur + sib).digest()
                if idx % 2 == 0
                else hashlib.sha256(sib + cur).digest()
            )
            idx >>= 1
        if not np.array_equal(np.frombuffer(node_hash, np.uint8), root):
            ok = False

    # Inverse: paths→octopus must reproduce the original multi-proof byte-for-byte.
    octopus = merkle.paths_to_multi_proof(paths, num_leaves, positions)
    if octopus.shape != proof.shape or not np.array_equal(octopus, proof):
        ok = False

    print(f"octopus↔paths round-trip ({name}, n={num_leaves} q={len(positions)}): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ok = True
    ok = _check(16, 64, [3], "single") and ok
    ok = _check(16, 64, [0, 1], "sibling-pair") and ok
    ok = _check(16, 64, [2, 5, 11], "spread") and ok
    ok = _check(64, 32, [0, 1, 2, 3, 40, 41], "dense-clusters") and ok
    ok = _check(256, 16, [7, 7, 100, 255], "with-dup") and ok
    ok = _check(8, 16, [1, 2, 3, 4, 5, 6], "near-full") and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
