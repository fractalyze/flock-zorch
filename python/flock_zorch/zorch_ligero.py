"""flock-zorch's Ligero matrix commit, delegating the low-degree extension to
zorch's code-generic Reed-Solomon (fractalyze/flock-zorch#32).

flock's in-tree `ligerito.ligero_commit` hand-rolls the additive NTT in
`flock_zorch.ntt`. This reproduces the SAME L0 commitment by encoding through
`zorch.coding.ReedSolomon` over `binary_field_ghash` — which flock-zorch#11 /
zorch#393 made byte-identical to flock's additive NTT — then Merkle-committing
the codeword rows with flock's SHA-256 tree.

The split is deliberate and follows both repos' non-negotiables: the low-degree
extension (scheme-agnostic) comes from zorch, while the SHA-256 Merkle stays
flock-zorch's. flock hashes raw field bytes (byte-identity to flock-core), so the
commitment must use flock's *unmarked* SHA-256, not zorch's Poseidon2
`commit.merkle`. This is the first flock-zorch consumer of zorch's PCS layer; it
de-risks the GHASH instantiation at the commitment level. The recursive driver
(proof bytes) is validated separately by zorch's own code-generic round-trip.

GPU only: binary-field arithmetic is unlowered on this env's CPU PJRT path
(multiply returns 0, scatter fails to legalize `i128 -> field.bf<7, ghash>`),
while the additive-NTT encode lowers cleanly on the GPU backend.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from flock_zorch import merkle
from zorch.coding.reed_solomon import ReedSolomon

GHASH = jnp.dtype("binary_field_ghash")


def to_ghash(u64_pairs) -> Array:
    """uint64 `[..., 2]` (F128 `lo‖hi`) -> `binary_field_ghash` `[...]`.

    flock stores an F128 as 16 little-endian bytes (`lo` then `hi`), exactly
    `binary_field_ghash`'s byte layout, so the reinterpret is a bitcast."""
    return lax.bitcast_convert_type(jnp.asarray(u64_pairs, jnp.uint64), GHASH)


def from_ghash(x: Array) -> np.ndarray:
    """`binary_field_ghash` `[...]` -> uint64 `[..., 2]` (F128 `lo‖hi`)."""
    return np.asarray(lax.bitcast_convert_type(x, jnp.uint64))


def ligero_commit(
    poly_ghash: Array,
    log_msg_cols: int,
    log_num_interleaved: int,
    log_inv_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """zorch-backed twin of flock's `ligerito.ligero_commit`.

    `poly_ghash`: the level's witness as `binary_field_ghash` `[num_int *
    msg_cols]`, in flock's interleaved SoA layout `poly[col*num_int + lane]`.

    Returns `(codeword, tree)` byte-identical to flock's:
    `codeword` uint64 `[block_len*num_int, 2]` (SoA `mat[block_pos*num_int +
    lane]`), `tree` uint8 `[2*block_len-1, 32]`.
    """
    msg_cols = 1 << log_msg_cols
    num_int = 1 << log_num_interleaved
    blowup = 1 << log_inv_rate
    block_len = msg_cols * blowup

    # Per-lane messages: message_matrix[lane] = the msg_cols coefficients of lane
    # `lane`, i.e. poly reshaped to x[col, lane] then transposed to [lane, col].
    x = jnp.asarray(poly_ghash).reshape(msg_cols, num_int)  # x[col, lane]
    message_matrix = x.T  # [num_int, msg_cols]

    # RS low-degree-extends each row (message axis) — the additive NTT for a
    # binary field. zorch#393 pins this byte-identical to flock's encode.
    code = ReedSolomon(msg_cols, blowup, GHASH)
    codeword = code.encode(message_matrix)  # [num_int, block_len]

    # flock's SoA is position-major, lane-minor: mat[block_pos*num_int + lane].
    cw_soa = from_ghash(codeword.T).reshape(block_len * num_int, 2)
    # Each Merkle leaf is one codeword position = num_int F128 = num_int*16 bytes.
    leaves = cw_soa.reshape(block_len, num_int * 2).view(np.uint8)
    tree = merkle.merkle_tree(leaves)
    return cw_soa, tree
