"""Binary SHA-256 Merkle tree — byte-identical to flock's `merkle::merkle_root` /
`merkle_tree`, built on `zorch.commit.merkle.MerkleTree` (the scheme-agnostic
commit/fold machinery) with flock's byte-SHA-256 as leaf hasher and compressor.

flock's construction (no domain separation): each leaf hash = `SHA256(leaf_bytes)`,
each internal node = `SHA256(left ‖ right)` (64-byte preimage). zorch's binary
`_fold_scan` produces the same per-level digests with an O(1)-in-height traced
body (it compresses a full-width buffer each level and slices the live prefix —
extra hashes, cheaper trace; Merkle is <1% of PCS commit). The flat tree layout
(`tree[0..n]` leaf hashes, then each level up, root at `tree[2n-2]`) is the
concatenation of zorch's `digest_layers`.

The octopus multi-proof stays flock-side: the proof layout is flock's assembly.
"""
from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np

from zorch.commit.merkle import MerkleTree

from zorch.hash.sha256 import INITIAL_STATE, sha256_chain, U32


def _pad_device(msg, length: int):
    """Device SHA-256 pad: uint8 [B, length] -> uint32 [B, nblocks, 16] BE, all-fnp
    (no host round-trip) so Merkle nodes stay device-resident across levels. flock-
    local; `length` is static and the compression itself is zorch's `sha256_chain`."""
    b = msg.shape[0]
    bitlen = length * 8
    nblocks = (length + 8) // 64 + 1
    total = nblocks * 64
    padded = fnp.zeros((b, total), dtype=fnp.uint8)
    padded = padded.at[:, :length].set(msg)
    padded = padded.at[:, length].set(fnp.uint8(0x80))
    for i in range(8):  # 8-byte big-endian bit length at the tail (static bytes)
        padded = padded.at[:, total - 8 + i].set(fnp.uint8((bitlen >> (8 * (7 - i))) & 0xFF))
    words = padded.reshape(b, nblocks, 16, 4).astype(fnp.uint32)
    return (words[..., 0] << U32(24)) | (words[..., 1] << U32(16)) | (words[..., 2] << U32(8)) | words[..., 3]


def _digest(msgs, length: int):
    """Marked batched SHA-256: uint8 [B, length] -> uint8 [B, 32] (`zorch.sha256`)."""
    return sha256_chain(INITIAL_STATE, _pad_device(msgs, length))


class _Sha256LeafHasher:
    """`leaf_hasher` seam: `SHA256(leaf_bytes)`. `as_bytes` maps a batch of stored
    leaf rows to their uint8 SHA-256 preimage — identity here, an element-byte
    reinterpret in the GHASH subclass — so it is the one hook that varies with the
    leaf dtype and a single `_Sha256MerkleTree` serves both. Batched hashing runs
    through `_Sha256MerkleTree._hash_leaves`; `hash` is the single-row form the
    inherited reconstruct/verify path calls."""
    out = 32

    def as_bytes(self, matrix):
        return matrix

    def hash(self, row):
        b = self.as_bytes(row[None])
        return _digest(b, b.shape[1])[0]

    # Value equality for static jit-zone keys (zorch #214): param-free -> by type.
    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


class _GhashSha256LeafHasher(_Sha256LeafHasher):
    """Leaves are `binary_field_ghash` rows; the preimage is the raw lo‖hi LE
    element bytes (flock's leaf preimage). The uint8 bitcast is the one working
    device ghash→integer direction (ghash→uint64 returns zeros, zorch#399)."""

    def as_bytes(self, matrix):
        return frx.lax.bitcast_convert_type(matrix, fnp.uint8).reshape(matrix.shape[0], -1)


class _Sha256Compressor:
    """`compressor` seam: 2-to-1 `SHA256(left ‖ right)` (64-byte preimage) over
    32-byte digests."""
    arity = 2
    chunk = 32

    def compress(self, group):
        return _digest(group.reshape(1, 64), 64)[0]

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


class _Sha256MerkleTree(MerkleTree):
    """`MerkleTree` with whole levels hashed batch-native: SHA-256's block schedule
    reads the batch axis from the shape, so the base `vmap(single-hash)` would
    retrace the marker decomposition at the wrong rank — override the two batching
    hooks with the [B, L] contract `zorch.hash.sha256` is written for. Row-major
    only (both hooks hash rows); the leaf hasher's `as_bytes` picks the uint8
    preimage, so one class serves both the uint8 and GHASH codeword trees."""

    def _hash_leaves(self, matrix):
        rows = self._leaf_hasher.as_bytes(matrix)
        return _digest(rows, rows.shape[1])

    def _compress_groups(self, groups):
        return _digest(groups.reshape(groups.shape[0], 64), 64)


_TREE = _Sha256MerkleTree(_Sha256LeafHasher(), _Sha256Compressor())
GHASH_TREE = _Sha256MerkleTree(_GhashSha256LeafHasher(), _Sha256Compressor())


@frx.jit
def _root(leaves):
    return _TREE.commit(leaves)[0]


@frx.jit
def _tree(leaves):
    _, layers = _TREE.commit(leaves)
    return fnp.concatenate(layers, axis=0)  # [2*n_leaves - 1, 32], flock's layout


def merkle_root(leaves) -> np.ndarray:
    """32-byte Merkle root of `n_leaves` equal-sized leaves. uint8 [n_leaves, leaf_size]
    -> uint8 [32], byte-identical to flock. One jit (commit fold is a single scan)."""
    return np.asarray(_root(fnp.asarray(leaves, dtype=fnp.uint8)))


def merkle_tree(leaves) -> np.ndarray:
    """Full flat Merkle tree, byte-identical to flock's `merkle_tree` layout:
    `tree[0..n]` = leaf hashes (level k), then level k-1, …, root at `tree[2n-2]`.

    leaves: uint8 [n, leaf_size] (n a power of two). Returns uint8 [2n-1, 32].
    One jit (zorch digest_layers, concatenated)."""
    return np.asarray(_tree(fnp.asarray(leaves, dtype=fnp.uint8)))


def _octopus_levels(positions, num_leaves: int):
    """flock's octopus dedup schedule — the shared walk behind both octopus
    assemblers (`merkle_multi_proof` from a tree, `paths_to_multi_proof` from paths).

    Yields, per tree level (leaves→root), the sorted-left-to-right list of active
    groups `(p, paired)`: `p` is the group's lower active node index and `paired`
    is True when its sibling `p ^ 1` is also active (recomputed from below, no proof
    element) or False when the sibling is a distinct emitted digest. The active set
    halves each level (`p >> 1`, deduped); callers attach the digest source (a `tree`
    slice or a path entry) and emit only on `not paired`."""
    active = sorted({int(p) for p in positions})
    for _level in range(num_leaves.bit_length() - 1):
        groups, i, n = [], 0, len(active)
        while i < n:
            p = active[i]
            paired = i + 1 < n and active[i + 1] == (p ^ 1)
            groups.append((p, paired))
            i += 2 if paired else 1
        yield groups
        active = sorted({p >> 1 for p in active})


def merkle_multi_proof(tree: np.ndarray, num_leaves: int, positions) -> np.ndarray:
    """Octopus multi-proof, byte-identical to flock `merkle::merkle_multi_proof`.

    Emits, per level (leaves→root), the sibling `tree[level_start + (p^1)]` of each
    active node whose sibling is NOT itself active — the shared `_octopus_levels`
    dedup schedule sourced from the flat tree. Returns uint8 [num_siblings, 32]."""
    if len(positions) == 0 or num_leaves == 1:
        return np.zeros((0, 32), dtype=np.uint8)
    proof, level_start, level_len = [], 0, num_leaves
    for groups in _octopus_levels(positions, num_leaves):
        for p, paired in groups:
            if not paired:
                proof.append(tree[level_start + (p ^ 1)])
        level_start += level_len
        level_len >>= 1
    return np.stack(proof) if proof else np.zeros((0, 32), dtype=np.uint8)


def paths_to_multi_proof(paths: np.ndarray, num_leaves: int, positions) -> np.ndarray:
    """Assemble flock's octopus multi-proof from a zorch `Opening`'s per-query
    authentication paths + the sampled query positions, byte-identical to
    `merkle_multi_proof` (gated by the ligerito oracle tests' `merkle_proof` fields).
    This is the prover-side bridge from zorch's per-query openings to flock's
    deduped octopus wire.

    The deduplicated octopus layout is positional (which siblings are emitted depends
    on which nodes are co-active), so it is not recoverable from the paths' shape
    alone — but every sibling it emits IS one path entry: query `qi` at leaf
    `positions[qi]` carries, at level L, `paths[qi, L]` = the digest of node
    `(positions[qi] >> L) ^ 1`, exactly the sibling flock emits for an active node
    whose sibling is not itself active. So this walks the shared `_octopus_levels`
    schedule, sourcing each emission from the paths — no tree rebuild.

    `paths`: uint8 [Q, depth, 32] (query-major, `np.stack(opening.path, axis=1)`);
    `positions`: length-Q query leaf indices (dups allowed). Returns uint8
    [num_siblings, 32]."""
    positions = [int(p) for p in positions]
    if not positions or num_leaves == 1:
        return np.zeros((0, 32), np.uint8)
    paths = np.asarray(paths)
    proof = []
    for level, groups in enumerate(_octopus_levels(positions, num_leaves)):
        node_to_qi = {}
        for qi, leaf in enumerate(positions):
            node_to_qi.setdefault(leaf >> level, qi)  # any query passing through this node
        for p, paired in groups:
            if not paired:
                proof.append(paths[node_to_qi[p], level])  # digest of node p^1
    return np.stack(proof) if proof else np.zeros((0, 32), np.uint8)
