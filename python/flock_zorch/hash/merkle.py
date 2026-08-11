"""Binary SHA-256 Merkle tree — byte-identical to flock's `merkle` module, built
on `zorch.commit.merkle.MerkleTree` (the scheme-agnostic commit/fold machinery)
with flock's byte-SHA-256 as leaf hasher and compressor.

flock's construction (no domain separation): each leaf hash = `SHA256(leaf_bytes)`,
each internal node = `SHA256(left ‖ right)` (64-byte preimage). zorch's binary
`_fold_scan` produces the same per-level digests with an O(1)-in-height traced
body (it compresses a full-width buffer each level and slices the live prefix —
extra hashes, cheaper trace; Merkle is <1% of PCS commit).

The octopus multi-proof stays flock-side: the proof layout is flock's assembly.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from hash_frx.sha256 import INITIAL_STATE, block_to_words, sha256_merkle_damgard
from zorch.commit.merkle import MerkleTree


def _pad_device(msg, length: int):
    """Device SHA-256 pad: uint8 [B, length] -> uint32 [B, nblocks, 16] BE, all-fnp
    (no host round-trip) so Merkle nodes stay device-resident across levels. flock-
    local; `length` is static and the compression itself is hash-frx's
    `sha256_merkle_damgard`.

    `length` being static makes every byte past the message identical for all rows
    and known here, so the suffix is a host constant broadcast onto the batch.
    Writing it instead with `.at[].set()` emits a `dynamic-update-slice` per write
    whose in-bounds guard XLA does not fold, and fusing that chain into the leaf
    transpose costs XLA its tiled-transpose emitter — see fractalyze/flock-zorch#205
    for the measurements."""
    b = msg.shape[0]
    # +8 for the length field and +1 block so the 0x80 never lands inside it.
    nblocks = (length + 8) // 64 + 1
    tail = np.zeros(nblocks * 64 - length, dtype=np.uint8)
    tail[0] = 0x80
    tail[-8:] = np.frombuffer((length * 8).to_bytes(8, "big"), np.uint8)
    padded = fnp.concatenate([msg, fnp.broadcast_to(tail, (b, tail.size))], axis=1)
    return block_to_words(padded)


def _digest(msgs, length: int):
    """Marked batched SHA-256: uint8 [B, length] -> uint8 [B, 32]
    (the `hash_frx.sha256` marker)."""
    return sha256_merkle_damgard(INITIAL_STATE, _pad_device(msgs, length))


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
        return frx.lax.bitcast_convert_type(matrix, fnp.uint8).reshape(
            matrix.shape[0], -1
        )


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
    hooks with the [B, L] contract `hash_frx.sha256` is written for. Row-major
    only (both hooks hash rows); the leaf hasher's `as_bytes` picks the uint8
    preimage, so one class serves both the uint8 and GHASH codeword trees."""

    def _hash_leaves(self, matrix):
        rows = self._leaf_hasher.as_bytes(matrix)
        return _digest(rows, rows.shape[1])

    def _compress_groups(self, groups):
        return _digest(groups.reshape(groups.shape[0], 64), 64)


GHASH_TREE = _Sha256MerkleTree(_GhashSha256LeafHasher(), _Sha256Compressor())


def paths_to_multi_proof(paths: np.ndarray, num_leaves: int, positions) -> np.ndarray:
    """Assemble flock's octopus multi-proof from a zorch `Opening`'s per-query
    authentication paths + the sampled query positions, byte-identical to flock
    `merkle::merkle_multi_proof` (gated by the Ligerito proof gates' `merkle_proof`
    fields). This is the prover-side bridge from zorch's per-query openings to
    flock's deduped octopus wire.

    The deduplicated octopus layout is positional (which siblings are emitted depends
    on which nodes are co-active), so it is not recoverable from the paths' shape
    alone — but every sibling it emits IS one path entry: query `qi` at leaf
    `positions[qi]` carries, at level L, `paths[qi, L]` = the digest of node
    `(positions[qi] >> L) ^ 1`, exactly the sibling flock emits for an active node
    whose sibling is not itself active. The active-node walk below sources each
    emission from the paths without rebuilding the tree.

    `paths`: uint8 [Q, depth, 32] (query-major, `np.stack(opening.path, axis=1)`);
    `positions`: length-Q query leaf indices (dups allowed). Returns uint8
    [num_siblings, 32]."""
    positions = np.asarray(positions, dtype=np.uint64).reshape(-1)
    if positions.size == 0 or num_leaves == 1:
        return np.zeros((0, 32), np.uint8)
    paths = np.asarray(paths)
    # Track each active node together with the first query whose path crosses
    # it.  Advancing this mapping with the tree avoids rebuilding a Python dict
    # from every original query at every level; NumPy performs the sort/dedup in
    # C and whole level slices are appended at once.
    active, first_qi = np.unique(positions, return_index=True)
    proof = []
    for level in range(num_leaves.bit_length() - 1):
        paired = np.zeros(active.size, dtype=np.bool_)
        paired[:-1] = (active[:-1] ^ np.uint64(1)) == active[1:]
        second_of_pair = np.zeros(active.size, dtype=np.bool_)
        second_of_pair[1:] = paired[:-1]
        emit = ~paired & ~second_of_pair
        if emit.any():
            proof.append(paths[first_qi[emit], level])

        # `active` is sorted, so the parents are non-decreasing and dedup is an
        # adjacent-difference scan. `np.unique` would argsort an already-ordered
        # array once per tree level — the dominant cost of the whole wire
        # assembly at m32 (six openings x ~16 levels).
        parents = active >> np.uint64(1)
        keep = np.empty(parents.size, np.bool_)
        keep[0] = True
        np.not_equal(parents[1:], parents[:-1], out=keep[1:])
        active = parents[keep]
        first_qi = first_qi[keep]

    return np.concatenate(proof, axis=0) if proof else np.zeros((0, 32), np.uint8)
