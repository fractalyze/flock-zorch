"""Unit test for `merkle.paths_to_multi_proof` — the prover-side assembler that
rebuilds flock's deduped octopus multi-proof from a zorch `Opening`'s per-query
authentication paths (what `pcs/ligerito.py` serializes into `merkle_proof`).

Pure host (hashlib reference tree + flock's own `merkle_multi_proof`), no GPU:
this is proof-format encoding, not field math. Builds each query's ground-truth
sibling path straight from the tree, checks it rebuilds the root, then asserts
paths→octopus reproduces flock's reference `merkle_multi_proof` byte-for-byte.
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np

from flock_zorch.hash import merkle


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
    # Ground-truth per-query paths straight from the tree: path[k] = sibling of (p>>k).
    paths = np.stack([
        np.stack([tree[starts[k] + ((p >> k) ^ 1)] for k in range(depth)])
        for p in positions
    ]) if depth else np.zeros((len(positions), 0, 32), np.uint8)

    ok = paths.shape == (len(positions), depth, 32)
    for qi, p in enumerate(positions):
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

    # paths→octopus must reproduce flock's reference multi-proof byte-for-byte.
    octopus = merkle.paths_to_multi_proof(paths, num_leaves, positions)
    if octopus.shape != proof.shape or not np.array_equal(octopus, proof):
        ok = False

    print(f"paths→octopus vs flock merkle_multi_proof ({name}, n={num_leaves} q={len(positions)}): "
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
