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
`2*2 = 4`), so the boundary is a bitcast, never `astype`. It is routed through
uint32 lanes: the direct `uint64[..,2] <-> ghash` bitcast silently miscompiles on
the CPU PJRT path (a fractalyze/xla BitcastConvertType bug), while `uint32[..,4]`
is correct on CPU and GPU and is the lane width zorch's kernels use natively.
Requires `jax_enable_x64`.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import lax
import zk_dtypes

from flock_zorch import field, sumcheck
from flock_zorch.challenger import Challenger
from zorch.pcs import ring_switch as zrs

U64 = jnp.uint64
GHASH = zk_dtypes.binary_field_ghash
LOG_PACKING = field.LOG_PACKING
LABEL = b"flock-ring-switch-v0"


def _to_ghash(arr):
    """`[.., 2]` uint64 (lo, hi) F128 -> `[..]` binary_field_ghash, byte-free."""
    arr = jnp.asarray(arr, U64)
    lanes = lax.bitcast_convert_type(arr, jnp.uint32).reshape(*arr.shape[:-1], 4)
    return lax.bitcast_convert_type(lanes, GHASH)


def _from_ghash(arr):
    """`[..]` binary_field_ghash -> `[.., 2]` uint64 (lo, hi). Inverse of _to_ghash."""
    lanes = lax.bitcast_convert_type(arr, jnp.uint32).reshape(*arr.shape, 2, 2)
    return np.asarray(lax.bitcast_convert_type(lanes, U64))


def _reduce_one(packed, x_outer, ch: Challenger, mul):
    """One claim's observe-and-reduce (the block prove and prove_batched share):
    observe LABEL + s_hat_v, sample r'', compute the sumcheck claim. Returns
    (s_hat_v [128,2], suffix_tensor [ghash], eq_r_dprime [128,2 lanes], claim [2]);
    the caller turns eq_r_dprime into rs_eq_ind (with or without a gamma scale)."""
    ch.observe_label(LABEL)
    suffix = jnp.asarray(np.asarray(x_outer)[1:])             # x_outer[1:], length L
    suffix_tensor = _to_ghash(sumcheck.build_eq_fused(suffix, mul=mul))
    s_hat_v = zrs.bit_slice_evals(packed, suffix_tensor)     # (128,) ghash
    s_hat_v_lanes = _from_ghash(s_hat_v)                     # [128,2]
    ch.observe_f128_slice(s_hat_v_lanes)
    r_dprime = jnp.asarray(ch.sample_f128_vec(LOG_PACKING))  # [7,2]
    eq_r_dprime = sumcheck.build_eq_fused(r_dprime, mul=mul)  # [128,2], kept in lanes for gamma
    claim = zrs.inner_product(zrs.tensor_algebra_transpose(s_hat_v), _to_ghash(eq_r_dprime))
    return s_hat_v_lanes, suffix_tensor, eq_r_dprime, _from_ghash(claim)


def prove(packed_witness, x_outer, ch: Challenger, mul=field.mul):
    """Returns (s_hat_v [128,2], rs_eq_ind [2^L,2], sumcheck_claim [2]).
    Byte-identical to flock `ring_switch::prove`."""
    packed = _to_ghash(packed_witness)
    s_hat_v_lanes, suffix_tensor, eq_r_dprime, claim = _reduce_one(packed, x_outer, ch, mul)
    rs_eq_ind = zrs.rs_eq_ind(suffix_tensor, _to_ghash(eq_r_dprime))
    return s_hat_v_lanes, _from_ghash(rs_eq_ind), claim


def prove_batched(packed_witness, x_outers, ch: Challenger, mul=field.mul):
    """Batched ring-switch over N opening points — byte-identical to flock
    `ring_switch::prove_batched_padded_with_precomputed`.

    Transcript: per claim (in order) observe `flock-ring-switch-v0` + s_hat_v +
    sample r_dprime[7]; THEN sample N gamma's (sound only after all observations);
    THEN bake gamma_i into each `rs_eq_ind_i` (the caller-owned linear combination
    — see the zorch module's contract). Returns
    (s_hat_vs, rs_eq_inds[gamma-baked], sumcheck_claims, gammas)."""
    packed = _to_ghash(packed_witness)
    works = [_reduce_one(packed, x_outer, ch, mul) for x_outer in x_outers]
    gammas = [ch.sample_f128() for _ in range(len(x_outers))]

    s_hat_vs, rs_eq_inds, sumcheck_claims = [], [], []
    for (s_hat_v_lanes, suffix_tensor, eq_r_dprime, claim), g in zip(works, gammas):
        scaled = _to_ghash(jnp.asarray(g)) * _to_ghash(jnp.asarray(eq_r_dprime))  # gamma baked into eq
        rs_eq_inds.append(_from_ghash(zrs.rs_eq_ind(suffix_tensor, scaled)))
        s_hat_vs.append(s_hat_v_lanes)
        sumcheck_claims.append(claim)
    return s_hat_vs, rs_eq_inds, sumcheck_claims, gammas
