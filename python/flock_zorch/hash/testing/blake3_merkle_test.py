# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Fork-fixture byte gate for the BLAKE3 Merkle arm.

The fixture below was printed by the flock-challenge fork's own `merkle`
module (Layr-Labs/flock-challenge @ d86604387a795f74aa526176f0911763df256405 —
the crate the benchmark harness verifies with): `hash_leaf` / `hash_pair` /
`merkle_root` over deterministic pattern bytes (`byte i = (i*7 + 3) & 0xff`),
under both hash kinds. To regenerate, call those three public functions from
a scratch bin path-depping `crates/flock-core` and paste the hex here.

What the BLAKE3 pins establish, in dependency order:

- **Leaf = non-root chaining value**, at every shape class a leaf can take:
  sub-block (16 B), non-block-aligned (48 B), exactly one block (64 B), one
  whole chunk — flock's L0 leaf — (1024 B), and multi-chunk (2080 B, a tree
  inside the leaf hash). A `blake3::hash`-shaped implementation (root-flagged)
  fails every one of these.
- **Parent = non-root `PARENT` compression** of the two child values.
- **The composed tree root matches the fork's `merkle_root`** through zorch's
  `MerkleTree` fold, over ghash leaf rows (the codeword dtype).
- The single-row hooks (`hash` / `compress`) agree with the batched level
  hashes: the open/verify round-trip walks them against a batched commit.

The SHA-256 control runs the same fixtures through the existing arm — it
isolates a BLAKE3-arm bug from a fixture/pattern-mapping bug, and pins the
SHA arm's primitives against the fork directly.

All-CPU, python-native (no golden files).
"""
from __future__ import annotations

import numpy as np
from absl.testing import absltest

from flock_zorch.ghash import _lanes_to_ghash
from flock_zorch.hash.merkle import (
    GHASH_BLAKE3_TREE,
    GHASH_SHA256_TREE,
    _blake3_leaf_digest,
    _blake3_parent_digest,
    sha256_digest,
)

_FIX_BLAKE3 = {
    "leaf16": "2f863e73db09bc4a63698045d77a8d5162a4c739984d11f90e084e70b451837d",
    "leaf48": "3112728f91d983e605af3c0079bcaf6afded1744568478d2f49a1899c38973a6",
    "leaf64": "fbe079c66d0b1a11a35d294dab36131ba3843db2e6a05ad977bdc8e694af2766",
    "leaf1024": "fc24c424932fe64966ccb7d366c151e9b5f9191eef6fb9b8597474b995817750",
    "leaf2080": "9755fb6365dee1fb244f630536667659e831d62f92a8e7fbdbc3c15a809c6bc4",
    "pair": "274e0f18ff632c3e77cd70a66eeef14729bc0453cfd27f6219910518b29103a7",
    "root8x48": "2639215170ca40008ba57669d3956e501e6e6b744095c9ea135a55f988f4c80f",
    "root4x1024": "7c069c0b12bc2b0d7f9063c21fe67b39c6a1c01a6e7c0b7c99d888d872e322df",
}
_FIX_SHA256 = {
    "leaf16": "9c94926dfb94433e790f2c209e2633b2dd3e922b2741ac687e164d488d1ff67c",
    "leaf48": "31cedec8e83dc0fb13e8ba27dfd62dd11aefa1923d78bfbade0eb4f339636144",
    "leaf64": "39e3d7b6b5d075d37d053ad89b24b41bef4f3c29760c84447cab3f3be1882241",
    "leaf1024": "e9183d9a79aad8a047b8e67981210d50b01fc75b1edba5bc32ba3d3ec4d5056d",
    "leaf2080": "c8296e74766acc7a041bdb6ac362cce459e3a708e9e9c4741ab4d99b55f033c8",
    "pair": "70d0f490dd71998c23c6cb02d6cd8daca2f3271af50a6d8523f045be5dbe6840",
    "root8x48": "5b112c9b7f6f61b4a669166461a6cd05181aa70be7c93b77e42f922de920f668",
    "root4x1024": "d46af8950f44fe0840e7a68bfcbf7507605044b8908f6cdee1cba1a95da91f22",
}

_LEAF_LENGTHS = (16, 48, 64, 1024, 2080)


def _pattern(n: int) -> bytes:
    return bytes((i * 7 + 3) & 0xFF for i in range(n))


def _row(data: bytes) -> np.ndarray:
    return np.frombuffer(data, np.uint8)[None, :]


def _hex(digest) -> str:
    return np.asarray(digest).tobytes().hex()


def _ghash_matrix(data: bytes, leaves: int) -> np.ndarray:
    """Pattern bytes as a `[leaves, elems]` ghash matrix whose element bytes
    are exactly `data` in row order — the tree's leaf preimage."""
    return _lanes_to_ghash(np.frombuffer(data, np.uint64).reshape(leaves, -1, 2))


def _blake3_pair(a_len: int, b_len: int):
    a = np.asarray(_blake3_leaf_digest(_row(_pattern(a_len)))[0])
    b = np.asarray(_blake3_leaf_digest(_row(_pattern(b_len)))[0])
    return _blake3_parent_digest(np.concatenate([a, b])[None, :])[0]


def _sha256_pair(a_len: int, b_len: int):
    a = np.asarray(sha256_digest(_row(_pattern(a_len)))[0])
    b = np.asarray(sha256_digest(_row(_pattern(b_len)))[0])
    return sha256_digest(np.concatenate([a, b])[None, :])[0]


class Blake3MerkleForkGateTest(absltest.TestCase):
    def test_leaf_digests_match_the_fork(self):
        for length in _LEAF_LENGTHS:
            got = _hex(_blake3_leaf_digest(_row(_pattern(length)))[0])
            self.assertEqual(got, _FIX_BLAKE3[f"leaf{length}"], f"leaf length {length}")

    def test_pair_digest_matches_the_fork(self):
        self.assertEqual(_hex(_blake3_pair(32, 96)), _FIX_BLAKE3["pair"])

    def test_tree_roots_match_the_fork(self):
        for key, leaves, width in (("root8x48", 8, 48), ("root4x1024", 4, 1024)):
            matrix = _ghash_matrix(_pattern(leaves * width), leaves)
            root, _ = GHASH_BLAKE3_TREE.commit(matrix)
            self.assertEqual(_hex(root), _FIX_BLAKE3[key], key)

    def test_open_verify_roundtrip(self):
        # Walks the single-row hooks (leaf `hash`, pair `compress`) against the
        # batched commit — the two must agree for openings to verify.
        matrix = _ghash_matrix(_pattern(8 * 48), 8)
        root, layers = GHASH_BLAKE3_TREE.commit(matrix)
        opening = GHASH_BLAKE3_TREE.open(matrix, layers, 3)
        self.assertTrue(bool(GHASH_BLAKE3_TREE.verify(root, 3, opening)))
        tampered = np.asarray(root).copy()
        tampered[0] ^= 1
        self.assertFalse(bool(GHASH_BLAKE3_TREE.verify(tampered, 3, opening)))


class Sha256MerkleControlTest(absltest.TestCase):
    """The existing SHA-256 arm through the same fixtures. A red BLAKE3 gate
    with this green is a BLAKE3-arm bug; both red is a pattern-mapping bug."""

    def test_leaf_digests_match_the_fork(self):
        for length in _LEAF_LENGTHS:
            got = _hex(sha256_digest(_row(_pattern(length)))[0])
            self.assertEqual(got, _FIX_SHA256[f"leaf{length}"], f"leaf length {length}")

    def test_pair_digest_matches_the_fork(self):
        self.assertEqual(_hex(_sha256_pair(32, 96)), _FIX_SHA256["pair"])

    def test_tree_roots_match_the_fork(self):
        for key, leaves, width in (("root8x48", 8, 48), ("root4x1024", 4, 1024)):
            matrix = _ghash_matrix(_pattern(leaves * width), leaves)
            root, _ = GHASH_SHA256_TREE.commit(matrix)
            self.assertEqual(_hex(root), _FIX_SHA256[key], key)


if __name__ == "__main__":
    absltest.main()
