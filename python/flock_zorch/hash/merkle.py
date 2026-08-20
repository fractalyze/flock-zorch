"""Binary Merkle tree, SHA-256 and BLAKE3 arms — byte-identical to flock's
`merkle` module, built on `zorch.commit.merkle.MerkleTree` (the scheme-agnostic
commit/fold machinery) with flock's leaf hasher and compressor injected.

The SHA-256 arm is flock's default construction (no domain separation): each
leaf hash = `SHA256(leaf_bytes)`, each internal node = `SHA256(left ‖ right)`
(64-byte preimage). zorch's binary `_fold_scan` produces the same per-level
digests with an O(1)-in-height traced body (it compresses a full-width buffer
each level and slices the live prefix — extra hashes, cheaper trace; Merkle is
<1% of PCS commit).

The BLAKE3 arm is the flock-challenge fork's `merkle_hash = Blake3`
(the benchmark profile's commitment), and it does NOT hash concatenated bytes:
it uses BLAKE3's own tree semantics — a leaf is the NON-ROOT chaining value of
the leaf bytes (`Hasher::update(bytes).finalize_non_root()`), a parent is one
`PARENT`-flagged compression of the two child values without the `ROOT` flag
(`merge_subtrees_non_root(l, r, Mode::Hash)`). The `PARENT` flag buys the
leaf/parent domain separation the SHA-256 construction lacks — and the digests
differ from a `blake3::hash`-per-node tree (root-flagged one-shot hashes), so
the arm must match the fork's choice exactly, not merely "BLAKE3". Both hooks
are hash-frx's pending-compression surface (`tree_output` / `parent_output`
finished by `chaining_value`), byte-gated against fork-dumped fixtures by
`testing/blake3_merkle_test.py`.

The octopus multi-proof stays flock-side: the proof layout is flock's assembly.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from hash_frx.blake3 import blake3
from hash_frx.sha256 import digest as sha256_digest
from zorch.commit.merkle import MerkleTree


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
        return sha256_digest(self.as_bytes(row[None]))[0]

    # Value equality for static jit-zone keys (zorch #214): param-free -> by type.
    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


def _ghash_leaf_bytes(matrix):
    """`binary_field_ghash` leaf rows -> their raw lo‖hi LE element bytes
    (flock's leaf preimage), uint8 `[B, 16*elems]`. The uint8 bitcast is the one
    working device ghash→integer direction (ghash→uint64 returns zeros,
    zorch#399). Shared by both hash arms — the preimage is the leaf dtype's,
    not the hash's."""
    return frx.lax.bitcast_convert_type(matrix, fnp.uint8).reshape(matrix.shape[0], -1)


class _GhashSha256LeafHasher(_Sha256LeafHasher):
    """Leaves are `binary_field_ghash` rows hashed over their element bytes."""

    def as_bytes(self, matrix):
        return _ghash_leaf_bytes(matrix)


class _Sha256Compressor:
    """`compressor` seam: 2-to-1 `SHA256(left ‖ right)` (64-byte preimage) over
    32-byte digests."""

    arity = 2
    chunk = 32

    def compress(self, group):
        return sha256_digest(group.reshape(1, 64))[0]

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
        return sha256_digest(self._leaf_hasher.as_bytes(matrix))

    def _compress_groups(self, groups):
        return sha256_digest(groups.reshape(groups.shape[0], 64))


GHASH_SHA256_TREE = _Sha256MerkleTree(_GhashSha256LeafHasher(), _Sha256Compressor())


# --- BLAKE3 arm (the flock-challenge fork's `merkle_hash = Blake3`) ----------
#
# Every hook is a pending BLAKE3 compression finished WITHOUT the `ROOT` flag —
# `chaining_value` — which is exactly the fork's `finalize_non_root` /
# `merge_subtrees_non_root` contract (see the module docstring). The digest
# bytes are the eight chaining-value words little-endian, the same packing the
# standard reads the root digest with.


def _blake3_leaf_digest(rows):
    """Non-root BLAKE3 chaining value per leaf: uint8 `[B, L]` -> uint8 `[B, 32]`.

    Marked, so the whole leaf level is one `hash_frx.blake3` composite carrying
    `non_root = 1` and fuses to a single kernel. Both backends this runs on have
    an emitter for it — GPU since fractalyze/xla#499, CPU since #507 — so the
    fallback this used to keep for the CPU leg is gone.

    That fallback was never about bytes, which are identical either way. It was
    the compile cliff an unrecognized marker pays: the composite inlines and
    XLA's codegen of the inlined body is super-linear in the compression count,
    so a chunk-sized leaf did not finish compiling, where the plain unrolled
    primitives compile in seconds at the same shapes. With the emitter present
    the body is never codegen'd at all — a chunk-sized leaf compiles in ~1s.

    A backend without a BLAKE3 emitter would pay that cliff again rather than
    fall back. None is in use here; the seam to restore if one appears is
    `blake3.unmarked_non_root_hash`, which is the same arithmetic unmarked.
    """
    return blake3.non_root_digest(rows)


def _blake3_parent_digest(pairs):
    """Non-root `PARENT` compression per node pair: uint8 `[G, 64]`
    (left ‖ right child digests) -> uint8 `[G, 32]`.

    Marked on every backend, unlike `_blake3_leaf_digest`. That one gates on the
    emitter because without one its marker inlines a chunk-sized body whose
    codegen is super-linear in the compression count; a parent is exactly ONE
    compression at every tree depth, so the inlined form is the same size as the
    primitives it replaces and there is no cliff to dodge. With an emitter a
    whole parent level is one kernel; without one it inlines to the arithmetic
    this function used to spell by hand. Same bytes either way — the pair gate
    in `blake3_merkle_test` pins them against the fork's own fixtures.

    The `pack_le`/`unpack_le` round-trip the hand-spelled form needed is gone:
    the two children ARE the marker's 64 operand bytes, so the packing folds
    into the emitter's own block read.
    """
    return blake3.parent_digest(pairs)


class _Blake3LeafHasher:
    """`leaf_hasher` seam for the BLAKE3 arm: the leaf bytes' non-root chaining
    value. Same `as_bytes` hook contract as the SHA-256 arm — batched hashing
    runs through `_Blake3MerkleTree._hash_leaves`; `hash` is the single-row form
    the inherited reconstruct/verify path calls."""

    out = 32

    def as_bytes(self, matrix):
        return matrix

    def hash(self, row):
        b = self.as_bytes(row[None])
        return _blake3_leaf_digest(b)[0]

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


class _GhashBlake3LeafHasher(_Blake3LeafHasher):
    """Leaves are `binary_field_ghash` rows hashed over their element bytes."""

    def as_bytes(self, matrix):
        return _ghash_leaf_bytes(matrix)


class _Blake3Compressor:
    """`compressor` seam: 2-to-1 non-root `PARENT` compression over 32-byte
    child chaining values."""

    arity = 2
    chunk = 32

    def compress(self, group):
        return _blake3_parent_digest(group.reshape(1, 64))[0]

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


class _Blake3MerkleTree(MerkleTree):
    """`MerkleTree` with whole levels hashed batch-native, like
    `_Sha256MerkleTree`: hash-frx's BLAKE3 reads the batch axis from the shape,
    so the base `vmap(single-hash)` would retrace the block schedule at the
    wrong rank. No dedicated fusion marker exists for BLAKE3 yet, so the traced
    compressions run de-fused — correct first; the benchmark profile's commit
    perf is a later lane."""

    def _hash_leaves(self, matrix):
        rows = self._leaf_hasher.as_bytes(matrix)
        return _blake3_leaf_digest(rows)

    def _compress_groups(self, groups):
        return _blake3_parent_digest(groups.reshape(groups.shape[0], 64))


GHASH_BLAKE3_TREE = _Blake3MerkleTree(_GhashBlake3LeafHasher(), _Blake3Compressor())


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
