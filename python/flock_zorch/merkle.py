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

The octopus multi-proof and the SHA-NI host path (`use_host_sha`) stay flock-side:
the proof layout is flock's assembly, and the host tree is one native call.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from zorch.commit.merkle import MerkleTree

from flock_zorch import sha256
from flock_zorch._merkle_host_ffi import (  # the SHA-NI off-GPU path (use_host_sha)
    host_sha_available, _merkle_tree_host, _merkle_root_host,
)


def _digest(msgs, length: int):
    """Marked batched SHA-256: uint8 [B, length] -> uint8 [B, 32] (`zorch.sha256`)."""
    return sha256._digest_words_marked(sha256._pad_device(msgs, length))


class _Sha256Leaf:
    """Leaf hasher seam value: `SHA256(leaf_bytes)`. Batched hashing goes through
    `_Sha256MerkleTree._hash_leaves`; this single-row form completes the seam
    contract for the inherited open/reconstruct paths (unexercised here yet)."""
    out = 32

    def hash(self, row):
        return _digest(row[None], row.shape[0])[0]

    # Value equality for static jit-zone keys (zorch #214): param-free -> by type.
    def __eq__(self, other):
        return isinstance(other, _Sha256Leaf)

    def __hash__(self):
        return hash(_Sha256Leaf)


class _Sha256Compress:
    """2-to-1 `SHA256(left ‖ right)` (64-byte preimage) over 32-byte digests."""
    arity = 2
    chunk = 32

    def compress(self, group):
        return _digest(group.reshape(1, 64), 64)[0]

    def __eq__(self, other):
        return isinstance(other, _Sha256Compress)

    def __hash__(self):
        return hash(_Sha256Compress)


class _Sha256MerkleTree(MerkleTree):
    """`MerkleTree` with whole levels hashed batch-native: SHA-256's block schedule
    reads the batch axis from the shape, so `vmap(single-hash)` would retrace the
    marker decomposition at the wrong rank — override the two batching hooks with
    the [B, L] contract `zorch.hash.sha256` is written for."""

    def __init__(self, leaf_hasher, compressor):
        # Row-major only: both hooks hash rows, ignoring the base column_major.
        super().__init__(leaf_hasher, compressor)

    def _hash_leaves(self, matrix):
        return _digest(matrix, matrix.shape[1])

    def _compress_groups(self, groups):
        return _digest(groups.reshape(groups.shape[0], 64), 64)


class _GhashSha256Leaf(_Sha256Leaf):
    """`_Sha256Leaf` over a `binary_field_ghash` row: hash the raw lo‖hi LE
    element bytes, flock's leaf preimage. The uint8 bitcast is the one working
    device ghash→integer direction (ghash→uint64 returns zeros, zorch#399)."""

    def hash(self, row):
        return super().hash(jax.lax.bitcast_convert_type(row, jnp.uint8).reshape(-1))

    def __eq__(self, other):
        return isinstance(other, _GhashSha256Leaf)

    def __hash__(self):
        return hash(_GhashSha256Leaf)


class _GhashSha256MerkleTree(_Sha256MerkleTree):
    """`_Sha256MerkleTree` whose leaves are `binary_field_ghash` rows instead of
    uint8 — the tree zorch's `commit_matrix` builds for a GHASH codeword (its
    `to_base_field` passes the 128-bit dtype through). Byte-identical to flock:
    each leaf hashes to `SHA256(row bytes)`, exactly `merkle_tree`'s preimage."""

    def _hash_leaves(self, matrix):
        u8 = jax.lax.bitcast_convert_type(matrix, jnp.uint8)
        return super()._hash_leaves(u8.reshape(matrix.shape[0], -1))


_TREE = _Sha256MerkleTree(_Sha256Leaf(), _Sha256Compress())
GHASH_TREE = _GhashSha256MerkleTree(_GhashSha256Leaf(), _Sha256Compress())


def verify_openings_flock(legs) -> bool:
    """`zorch.pcs.fold.verify_openings` over flock's SHA-256 Merkle tree: AND of
    "every opened leaf rebuilds its committed root" across `legs`
    (`(root, indices, Opening)`). The BaseFold verifier assembles legs by
    expanding flock's octopus proof (`multi_proof_to_paths`) into per-query
    `Opening`s. Returns a python bool."""
    from zorch.pcs.fold import verify_openings
    return bool(verify_openings(_TREE, legs))


@jax.jit
def _root_dev(leaves):
    return _TREE.commit(leaves)[0]


@jax.jit
def _tree_dev(leaves):
    _, layers = _TREE.commit(leaves)
    return jnp.concatenate(layers, axis=0)  # [2*n_leaves - 1, 32], flock's layout


def merkle_root(leaves, use_host_sha: bool = False) -> np.ndarray:
    """32-byte Merkle root of `n_leaves` equal-sized leaves. uint8 [n_leaves, leaf_size]
    -> uint8 [32], byte-identical to flock. One jit (commit fold is a single scan),
    or built on the host with flock's SHA-NI Merkle when `use_host_sha`."""
    if use_host_sha:
        return _merkle_root_host(leaves)
    return np.asarray(_root_dev(jnp.asarray(leaves, dtype=jnp.uint8)))


def merkle_root_from_flat(data, n_leaves: int) -> np.ndarray:
    """Convenience: split a flat uint8 buffer into `n_leaves` equal leaves, hash."""
    data = np.asarray(data, dtype=np.uint8).reshape(n_leaves, -1)
    return merkle_root(data)


def merkle_tree(leaves, use_host_sha: bool = False) -> np.ndarray:
    """Full flat Merkle tree, byte-identical to flock's `merkle_tree` layout:
    `tree[0..n]` = leaf hashes (level k), then level k-1, …, root at `tree[2n-2]`.

    leaves: uint8 [n, leaf_size] (n a power of two). Returns uint8 [2n-1, 32].
    One jit (zorch digest_layers, concatenated), or built on the host with
    flock's SHA-NI Merkle when `use_host_sha`."""
    if use_host_sha:
        return _merkle_tree_host(leaves)
    return np.asarray(_tree_dev(jnp.asarray(leaves, dtype=jnp.uint8)))


def merkle_multi_proof(tree: np.ndarray, num_leaves: int, positions) -> np.ndarray:
    """Octopus multi-proof, byte-identical to flock `merkle::merkle_multi_proof`.

    Emits, per level (leaves→root), the sibling `tree[level_start + (p^1)]` of each
    active node whose sibling is NOT itself active — sorted+deduped positions,
    bottom-up, left-to-right. Returns uint8 [num_siblings, 32]."""
    if len(positions) == 0 or num_leaves == 1:
        return np.zeros((0, 32), dtype=np.uint8)
    active = sorted(set(int(p) for p in positions))
    proof = []
    level_start, level_len = 0, num_leaves
    while level_len > 1:
        nxt, i = [], 0
        while i < len(active):
            p = active[i]
            if i + 1 < len(active) and active[i + 1] == (p ^ 1):
                i += 2                                   # sibling also active → no emit
            else:
                proof.append(tree[level_start + (p ^ 1)])
                i += 1
            nxt.append(p >> 1)
        active = nxt
        level_start += level_len
        level_len >>= 1
    return np.stack(proof) if proof else np.zeros((0, 32), dtype=np.uint8)


def _sha(*parts: bytes) -> bytes:
    import hashlib
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def multi_proof_to_paths(proof: np.ndarray, num_leaves: int, positions,
                         leaf_bytes: np.ndarray) -> np.ndarray:
    """Invert `merkle_multi_proof`: reconstruct each query's per-level sibling
    path from flock's octopus proof, so the BaseFold verifier can feed zorch's
    `pcs.fold.verify_openings` (which wants per-query `Opening(row, path)` rather
    than flock's shared/deduped wire — see `merkle.py` header: octopus is flock's
    proof assembly, kept host-side).

    `positions`: length-Q query leaf indices (dups allowed); `leaf_bytes`:
    uint8 [Q, leaf_len] the queried leaves aligned to `positions`. Returns
    `paths` uint8 [Q, depth, 32], leaf-first, `depth = log2(num_leaves)`.

    Replays flock's bottom-up walk, filling each active node's sibling from the
    next proof element (sibling inactive) or the co-active node's running hash
    (sibling active, computed from the level below). Roots are NOT trusted from
    this host walk — `verify_openings` independently rebuilds them on-device."""
    positions = [int(p) for p in positions]
    depth = num_leaves.bit_length() - 1
    q = len(positions)
    if depth == 0:
        return np.zeros((q, 0, 32), dtype=np.uint8)

    leaf_hash = {}
    for qi, p in enumerate(positions):
        leaf_hash.setdefault(p, _sha(leaf_bytes[qi].tobytes()))

    active = sorted(leaf_hash)
    cur = dict(leaf_hash)                 # node index -> running digest at this level
    sibling_at_level: list[dict] = []     # level k: node index -> its sibling digest
    pit = 0
    for _level in range(depth):
        sib, parents, i = {}, {}, 0
        while i < len(active):
            p = active[i]
            if i + 1 < len(active) and active[i + 1] == (p ^ 1):
                s = cur[p ^ 1]
                sib[p] = cur[p ^ 1]
                sib[p ^ 1] = cur[p]
                i += 2
            else:
                s = proof[pit]; pit += 1
                sib[p] = np.asarray(s, np.uint8)
                i += 1
            lo, hi = (cur[p], sib[p]) if p % 2 == 0 else (sib[p], cur[p])
            lo = lo if isinstance(lo, bytes) else lo.tobytes()
            hi = hi if isinstance(hi, bytes) else hi.tobytes()
            parents[p >> 1] = _sha(lo, hi)
        sibling_at_level.append(sib)
        active = sorted(parents)
        cur = parents

    paths = np.zeros((q, depth, 32), dtype=np.uint8)
    for qi, p in enumerate(positions):
        for k in range(depth):
            s = sibling_at_level[k][p >> k]
            paths[qi, k] = np.frombuffer(s, np.uint8) if isinstance(s, bytes) else s
    return paths
