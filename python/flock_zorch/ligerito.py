"""flock's Ligerito recursive PCS, authored in jax — byte-identical to flock-core
`pcs::ligerito`. The PCS flock's headline hash benchmarks use: instead of a single
codeword folded over FRI rounds (BaseFold), it RE-COMMITS the partially-folded
witness at R recursive levels (shrinking RS rate), driving one continuous
interleaved sumcheck (fold / introduce / glue) with per-level OOD binding, query
PoW, and a plaintext residual. Proof scales ~log²(log_n).

Reuses flock-zorch's byte-identical primitives: `ntt` (compute_twiddles +
forward_transform_interleaved), `merkle` (tree/multi_proof + host SHA-NI),
`sumcheck` (build_eq / fold / _prime), `challenger` (SHA-256 FS + grind_pow),
`ring_switch.prove_batched`. This module adds the recursive driver.

Port status (M0): `ligero_commit` (per-level RS-encode + Merkle). The SumcheckProver
driver, induce_sumcheck_poly + LCH basis, and the recursion loop land in M1..M7.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from flock_zorch import field, ntt as ntt_mod, merkle


def ligero_commit(poly, log_msg_cols: int, log_num_interleaved: int, log_inv_rate: int,
                  mul=field.mul, use_host_sha: bool = False):
    """Per-level Ligero commit (flock `pcs::ligerito::ligero_commit`): reshape `poly`
    (len num_interleaved·msg_cols, SoA `poly[col·num_interleaved + lane]`) into a
    block_len × num_interleaved matrix, RS-encode each lane via the additive NTT
    (zero-pad msg_cols→block_len then forward-transform — byte-identical to flock's
    replicate-+-start-at-layer form, as the first log_inv_rate layers are copies),
    and Merkle-commit the rows (leaf = num_interleaved F128). Returns (mat, tree).

    mat: uint64 [block_len·num_interleaved, 2] (SoA); tree: uint8 [2·block_len-1, 32]."""
    msg_cols = 1 << log_msg_cols
    num_int = 1 << log_num_interleaved
    block_len = msg_cols << log_inv_rate
    log_block_len = log_msg_cols + log_inv_rate

    x = jnp.asarray(poly).reshape(msg_cols, num_int, 2)
    pad = jnp.zeros((block_len - msg_cols, num_int, 2), dtype=x.dtype)
    cw = jnp.concatenate([x, pad], axis=0).reshape(block_len * num_int, 2)
    tw = jnp.asarray(ntt_mod.compute_twiddles(log_block_len))
    cw = ntt_mod.forward_transform_interleaved(cw, tw, log_block_len, num_int, mul=mul)
    mat = np.asarray(cw)
    leaves = mat.reshape(block_len, num_int * 2).view(np.uint8)
    tree = merkle.merkle_tree(leaves, use_host_sha=use_host_sha)
    return mat, tree
