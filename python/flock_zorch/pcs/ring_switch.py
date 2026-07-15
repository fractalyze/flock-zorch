"""flock's ring-switching reduction (`pcs::ring_switch::prove`) — a thin adapter
over the agnostic `zorch.pcs.ring_switch` DP24 block.

The bit->packed reduction kernels (bit-slice partial evals `s_hat_v`, the
transparent `rs_eq_ind`, the tensor-algebra transpose, the claim inner product)
live in zorch, dtype-native over `binary_field_ghash`. This module keeps only
what is flock-specific: the GHASH uint64-lane <-> `binary_field_ghash` boundary,
the Fiat-Shamir order (observe `flock-ring-switch-v0` + s_hat_v, sample r''), the
batched gamma combination, and the uint64-lane serialization the byte-gate reads.

flock's F128 is `uint64 [.., 2] = [lo, hi]` with bit i = coefficient of x^i — the
little-endian storage of `binary_field_ghash` (same GHASH basis, verified
`2*2 = 4`), so the boundary is a direct bitcast, never `astype`.
Requires `jax_enable_x64`.
"""
from __future__ import annotations

import numpy as np
import frx.numpy as jnp

from flock_zorch import field, sumcheck
from flock_zorch.challenger import Challenger
from zorch.pcs import ring_switch as zrs

LOG_PACKING = field.LOG_PACKING
LABEL = b"flock-ring-switch-v0"


def _reduce_one(packed, x_outer, ch: Challenger):
    """One claim's observe-and-reduce (the block prove and prove_batched share):
    observe LABEL + s_hat_v, sample r'', compute the sumcheck claim. Returns
    (s_hat_v [128,2], suffix_tensor [ghash], eq_r_dprime [128] ghash, claim [2]);
    the caller turns eq_r_dprime into rs_eq_ind (with or without a gamma scale)."""
    ch.observe_label(LABEL)
    suffix = jnp.asarray(np.asarray(x_outer)[1:])             # x_outer[1:], length L
    suffix_tensor = sumcheck.build_eq_fused_g(suffix)
    s_hat_v = zrs.bit_slice_evals(packed, suffix_tensor)     # (128,) ghash
    ch.observe_f128_slice_g(s_hat_v)                          # observe device ghash directly
    s_hat_v_lanes = field.from_ghash_host(s_hat_v)           # [128,2] — materialized for the proof only
    r_dprime = jnp.asarray(ch.sample_f128_vec(LOG_PACKING))  # [7,2]
    eq_r_dprime = sumcheck.build_eq_fused_g(r_dprime)  # [128] ghash, kept for the gamma combine
    claim = zrs.inner_product(zrs.tensor_algebra_transpose(s_hat_v), eq_r_dprime)
    return s_hat_v_lanes, suffix_tensor, eq_r_dprime, field.from_ghash_host(claim)


def prove(packed_witness, x_outer, ch: Challenger):
    """Returns (s_hat_v [128,2], rs_eq_ind [2^L,2], sumcheck_claim [2]).
    Byte-identical to flock `ring_switch::prove`."""
    packed = field.to_ghash(packed_witness)
    s_hat_v_lanes, suffix_tensor, eq_r_dprime, claim = _reduce_one(packed, x_outer, ch)
    rs_eq_ind = zrs.rs_eq_ind(suffix_tensor, eq_r_dprime)
    return s_hat_v_lanes, field.from_ghash_host(rs_eq_ind), claim


def prove_batched(packed_witness, x_outers, ch: Challenger):
    """Batched ring-switch over N opening points — byte-identical to flock
    `ring_switch::prove_batched_padded_with_precomputed`.

    Transcript: per claim (in order) observe `flock-ring-switch-v0` + s_hat_v +
    sample r_dprime[7]; THEN sample N gamma's (sound only after all observations);
    THEN bake gamma_i into each `rs_eq_ind_i` (the caller-owned linear combination
    — see the zorch module's contract). Returns
    (s_hat_vs, rs_eq_inds[gamma-baked], sumcheck_claims, gammas)."""
    packed = field.to_ghash(packed_witness)
    works = [_reduce_one(packed, x_outer, ch) for x_outer in x_outers]
    gammas = [ch.sample_f128() for _ in range(len(x_outers))]

    s_hat_vs, rs_eq_inds, sumcheck_claims = [], [], []
    for (s_hat_v_lanes, suffix_tensor, eq_r_dprime, claim), g in zip(works, gammas):
        scaled = field.to_ghash(jnp.asarray(g)) * eq_r_dprime  # gamma baked into eq
        rs_eq_inds.append(field.from_ghash_host(zrs.rs_eq_ind(suffix_tensor, scaled)))
        s_hat_vs.append(s_hat_v_lanes)
        sumcheck_claims.append(claim)
    return s_hat_vs, rs_eq_inds, sumcheck_claims, gammas
