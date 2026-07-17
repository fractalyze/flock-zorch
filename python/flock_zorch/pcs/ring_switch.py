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

import frx

from flock_zorch import fs, ghash, sumcheck
from flock_zorch.challenger import Challenger
from zorch.pcs import ring_switch as zrs

LOG_PACKING = ghash.LOG_PACKING
LABEL = b"flock-ring-switch-v0"


@frx.jit
def _reduce_one(t, packed, x_outer):
    """One claim's observe-and-reduce (the block prove and prove_batched share):
    observe LABEL + s_hat_v, sample r'', compute the sumcheck claim. A pure jitted
    region that THREADS the functional transcript `t` in and out — so the whole
    observe/build/sample/reduce fuses as one program. `build_eq` is called directly
    (no `build_eq_fused`: under this outer jit it already fuses), and the sampled
    challenge stays native ghash — no `from_ghash`→`to_ghash` lane round-trip.

    Returns (t, s_hat_v [128] ghash, suffix_tensor [ghash], eq_r_dprime [128] ghash,
    claim [ghash]); the caller turns eq_r_dprime into rs_eq_ind (with or without a
    gamma scale)."""
    t = fs.observe_label(t, LABEL)
    suffix = x_outer[1:]                                       # ghash coords, length L
    suffix_tensor = sumcheck.build_eq(suffix)
    s_hat_v = zrs.bit_slice_evals(packed, suffix_tensor)     # (128,) ghash
    t = fs.observe_slice(t, s_hat_v)                  # observe device ghash directly
    t, r_dprime = fs.sample_slice(t, LOG_PACKING)     # [7] ghash, kept native
    eq_r_dprime = sumcheck.build_eq(r_dprime)          # [128] ghash, for the gamma combine
    claim = zrs.inner_product(zrs.tensor_algebra_transpose(s_hat_v), eq_r_dprime)
    return t, s_hat_v, suffix_tensor, eq_r_dprime, claim       # claim native ghash


def prove(packed_witness, x_outer, ch: Challenger):
    """Returns (s_hat_v [128,2], rs_eq_ind [2^L] ghash, sumcheck_claim [ghash]).
    Byte-identical to flock `ring_switch::prove`."""
    packed = ghash.to_ghash(packed_witness)
    # Thread the mutable Challenger's functional transcript through the jitted region.
    ch._t, s_hat_v, suffix_tensor, eq_r_dprime, claim = _reduce_one(ch._t, packed, x_outer)
    rs_eq_ind = zrs.rs_eq_ind(suffix_tensor, eq_r_dprime)
    return s_hat_v, rs_eq_ind, claim                          # rs_eq_ind native ghash (the open's b)


def prove_batched(packed_witness, x_outers, ch: Challenger):
    """Batched ring-switch over N opening points — byte-identical to flock
    `ring_switch::prove_batched_padded_with_precomputed`.

    Transcript: per claim (in order) observe `flock-ring-switch-v0` + s_hat_v +
    sample r_dprime[7]; THEN sample N gamma's (sound only after all observations);
    THEN bake gamma_i into each `rs_eq_ind_i` (the caller-owned linear combination
    — see the zorch module's contract). Returns
    (s_hat_vs, rs_eq_inds[gamma-baked], sumcheck_claims, gammas)."""
    packed = ghash.to_ghash(packed_witness)
    works = []
    for x_outer in x_outers:
        ch._t, *work = _reduce_one(ch._t, packed, x_outer)
        works.append(work)                                    # [s_hat_v, suffix_tensor, eq_r_dprime, claim]
    gammas = [ch.sample_f128() for _ in range(len(x_outers))]

    s_hat_vs, rs_eq_inds, sumcheck_claims = [], [], []
    for (s_hat_v, suffix_tensor, eq_r_dprime, claim), g in zip(works, gammas):
        scaled = g * eq_r_dprime  # gamma baked into eq
        rs_eq_inds.append(zrs.rs_eq_ind(suffix_tensor, scaled))  # ghash [2^L], device-resident
        s_hat_vs.append(s_hat_v)
        sumcheck_claims.append(claim)
    return s_hat_vs, rs_eq_inds, sumcheck_claims, gammas
