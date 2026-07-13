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

import jax
import jax.numpy as jnp
import numpy as np

from zorch.commit.merkle import MerkleTree

from zorch.hash.sha256 import INITIAL_STATE, sha256_chain, U32


def _pad_device(msg, length: int):
    """Device SHA-256 pad: uint8 [B, length] -> uint32 [B, nblocks, 16] BE, all-jnp
    (no host round-trip) so Merkle nodes stay device-resident across levels. flock-
    local; `length` is static and the compression itself is zorch's `sha256_chain`."""
    b = msg.shape[0]
    bitlen = length * 8
    nblocks = (length + 8) // 64 + 1
    total = nblocks * 64
    padded = jnp.zeros((b, total), dtype=jnp.uint8)
    padded = padded.at[:, :length].set(msg)
    padded = padded.at[:, length].set(jnp.uint8(0x80))
    for i in range(8):  # 8-byte big-endian bit length at the tail (static bytes)
        padded = padded.at[:, total - 8 + i].set(jnp.uint8((bitlen >> (8 * (7 - i))) & 0xFF))
    words = padded.reshape(b, nblocks, 16, 4).astype(jnp.uint32)
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
        return jax.lax.bitcast_convert_type(matrix, jnp.uint8).reshape(matrix.shape[0], -1)


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


def verify_openings(legs) -> bool:
    """`zorch.pcs.fold.verify_openings` over flock's SHA-256 Merkle tree: AND of
    "every opened leaf rebuilds its committed root" across `legs`
    (`(root, indices, Opening)`). The BaseFold verifier assembles legs by
    expanding flock's octopus proof (`multi_proof_to_paths`) into per-query
    `Opening`s. Returns a python bool."""
    from zorch.pcs.fold import verify_openings as _verify_openings
    return bool(_verify_openings(_TREE, legs))


@jax.jit
def _root_dev(leaves):
    return _TREE.commit(leaves)[0]


@jax.jit
def _tree_dev(leaves):
    _, layers = _TREE.commit(leaves)
    return jnp.concatenate(layers, axis=0)  # [2*n_leaves - 1, 32], flock's layout


def merkle_root(leaves) -> np.ndarray:
    """32-byte Merkle root of `n_leaves` equal-sized leaves. uint8 [n_leaves, leaf_size]
    -> uint8 [32], byte-identical to flock. One jit (commit fold is a single scan)."""
    return np.asarray(_root_dev(jnp.asarray(leaves, dtype=jnp.uint8)))


def merkle_tree(leaves) -> np.ndarray:
    """Full flat Merkle tree, byte-identical to flock's `merkle_tree` layout:
    `tree[0..n]` = leaf hashes (level k), then level k-1, …, root at `tree[2n-2]`.

    leaves: uint8 [n, leaf_size] (n a power of two). Returns uint8 [2n-1, 32].
    One jit (zorch digest_layers, concatenated)."""
    return np.asarray(_tree_dev(jnp.asarray(leaves, dtype=jnp.uint8)))


def _octopus_levels(positions, num_leaves: int):
    """flock's octopus dedup schedule — the shared walk behind all three octopus
    assemblers (`merkle_multi_proof`, `multi_proof_to_paths`, `paths_to_multi_proof`).

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

    cur = dict(leaf_hash)                 # node index -> running digest at this level
    sibling_at_level: list[dict] = []     # level k: node index -> its sibling digest
    pit = 0
    for groups in _octopus_levels(positions, num_leaves):
        sib, parents = {}, {}
        for p, paired in groups:
            if paired:                    # sibling co-active → recomputed from below
                sib[p] = cur[p ^ 1]
                sib[p ^ 1] = cur[p]
            else:                         # sibling inactive → next proof element
                sib[p] = np.asarray(proof[pit], np.uint8); pit += 1
            lo, hi = (cur[p], sib[p]) if p % 2 == 0 else (sib[p], cur[p])
            lo = lo if isinstance(lo, bytes) else lo.tobytes()
            hi = hi if isinstance(hi, bytes) else hi.tobytes()
            parents[p >> 1] = _sha(lo, hi)
        sibling_at_level.append(sib)
        cur = parents

    paths = np.zeros((q, depth, 32), dtype=np.uint8)
    for qi, p in enumerate(positions):
        for k in range(depth):
            s = sibling_at_level[k][p >> k]
            paths[qi, k] = np.frombuffer(s, np.uint8) if isinstance(s, bytes) else s
    return paths


def paths_to_multi_proof(paths: np.ndarray, num_leaves: int, positions) -> np.ndarray:
    """Inverse of `multi_proof_to_paths`: assemble flock's octopus multi-proof from a
    zorch `Opening`'s per-query authentication paths + the sampled query positions,
    byte-identical to `merkle_multi_proof` (gated by the ligerito oracle tests'
    `merkle_proof` fields).

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
