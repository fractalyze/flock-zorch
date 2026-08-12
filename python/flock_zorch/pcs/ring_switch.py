"""flock's ring-switching reduction (`pcs::ring_switch::prove`) — a thin adapter
over the agnostic `zorch.pcs.ring_switch` DP24 block.

The bit->packed reduction kernels (bit-slice partial evals `s_hat_v`, the
transparent `rs_eq_ind`, the tensor-algebra transpose, the claim inner product)
live in zorch, dtype-native over `binary_field_ghash`. This module keeps only
what is flock-specific: the GHASH uint64-lane <-> `binary_field_ghash` boundary,
the Fiat-Shamir order (observe `flock-ring-switch-v0` + s_hat_v, sample r''), the
batched gamma combination, and the uint64-lane serialization the proof gates read.

flock's F128 is `uint64 [.., 2] = [lo, hi]` with bit i = coefficient of x^i — the
little-endian storage of `binary_field_ghash` (same GHASH basis, verified
`2*2 = 4`), so the boundary is a direct bitcast, never `astype`.
Requires `jax_enable_x64`.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
from zorch.pcs.ring_switch import bit_slice_evals, rs_eq_ind, tensor_algebra_transpose

from flock_zorch import fs, ghash, sumcheck
from flock_zorch.sha256_challenger import Sha256Challenger

LOG_PACKING = ghash.LOG_PACKING
LABEL = b"flock-ring-switch-v0"


def _inner_product(a, b):
    return fnp.sum(a * b, axis=0)


@frx.jit
def _slice_evals(packed, suffixes):
    """The transcript-independent half of the batched open, one read per claim.

    Builds each opening point's suffix eq tensor and bit-slices the witness per
    claim. Per-claim beats the shared batched read on both of its costs: the
    single-claim elements kernel takes the two-row tile the batched kernel
    cannot (its batch axis already owns the registers), and the batched form
    additionally materializes the `(2^L, N)` suffix stack — glue that costs
    more at phase level than the shared witness read saves. `suffixes` is a
    tuple of N `[L]` ghash coord vectors (static N); the build stays under this
    jit so `build_eq` fuses. Returns (s_hat_vs: N x `[128]` ghash,
    suffix_tensors: N contiguous `[2^L]` ghash — `rs_eq_ind` reads each
    contiguously)."""
    suffix_tensors = [sumcheck.build_eq(s) for s in suffixes]  # N x (2^L,) ghash
    s_hat_vs = [bit_slice_evals(packed, t) for t in suffix_tensors]
    return s_hat_vs, suffix_tensors


@frx.jit
def _observe_and_reduce(t, s_hat_v):
    """One claim's serial Fiat-Shamir with `s_hat_v` already computed: observe
    LABEL + s_hat_v, sample r'', build the reduced sumcheck claim. THREADS the
    functional transcript `t` in unchanged order (observe_label → observe_slice →
    sample_slice), so lifting the transcript-free `bit_slice_evals` out into the
    batched pass above leaves every Fiat-Shamir draw byte-identical to the old
    per-claim loop. Returns (t, eq_r_dprime [128] ghash, claim [ghash])."""
    t = fs.observe_label(t, LABEL)
    t = fs.observe_slice(t, s_hat_v)  # observe device ghash directly
    t, r_dprime = fs.sample_slice(t, LOG_PACKING)  # [7] ghash, kept native
    eq_r_dprime = sumcheck.build_eq(r_dprime)  # [128] ghash, for the gamma combine
    claim = _inner_product(tensor_algebra_transpose(s_hat_v), eq_r_dprime)
    return t, eq_r_dprime, claim  # claim native ghash


def prove_batched(packed_witness, x_outers, ch: Sha256Challenger):
    """Batched ring-switch over N opening points — byte-identical to flock
    `ring_switch::prove_batched_padded_with_precomputed`.

    The transcript-free witness reduction (`_slice_evals`) is lifted out of the
    serial per-claim Fiat-Shamir (`_observe_and_reduce`), which runs in the same
    order as before, so the wire is unchanged.

    Transcript: per claim (in order) observe `flock-ring-switch-v0` + s_hat_v +
    sample r_dprime[7]; THEN sample N gamma's (sound only after all observations);
    THEN bake gamma_i into each `rs_eq_ind_i` (the caller-owned linear combination
    — see the zorch module's contract). Returns
    (s_hat_vs, rs_eq_inds[gamma-baked], sumcheck_claims, gammas)."""
    packed = ghash.to_ghash(packed_witness)
    suffixes = tuple(x_outer[1:] for x_outer in x_outers)  # ghash coords, length L
    s_hat_vs, suffix_tensors = _slice_evals(packed, suffixes)

    eq_r_dprimes, claims = [], []
    for i in range(len(x_outers)):
        ch._t, eq_r_dprime, claim = _observe_and_reduce(ch._t, s_hat_vs[i])
        eq_r_dprimes.append(eq_r_dprime)
        claims.append(claim)
    gammas = [ch.sample_f128() for _ in range(len(x_outers))]

    # Bake gamma_i into each eq, then rs_eq_ind it against the claim's contiguous
    # suffix tensor -> ghash [2^L], device-resident (the caller-owned combination).
    rs_eq_inds = [
        rs_eq_ind(suffix_tensors[i], g * eq_r_dprimes[i]) for i, g in enumerate(gammas)
    ]
    return list(s_hat_vs), rs_eq_inds, claims, gammas


# ---- verifier side ---------------------------------------------------------

from flock_zorch.zerocheck import _lagrange_weights  # noqa: E402

_ONE_G = ghash.to_ghash(fnp.array([1, 0], fnp.uint64))
_CLAIM_K = LOG_PACKING - 1  # 6: the φ8 skip dim; bit-6 carries the x_outer[0] eq split


def _build_claim_weights(z_skip, x_outer_0):
    """flock `ring_switch::build_claim_weights`: lam(z_skip) ⊗ eq(x_outer[0]).
    i∈[0,64) take eq(x0,0)=1+x0; i∈[64,128) take eq(x0,1)=x0."""
    lam = _lagrange_weights(_CLAIM_K, z_skip, 0)  # [64] ghash, φ8 S-domain
    return fnp.concatenate([lam * (_ONE_G + x_outer_0), lam * x_outer_0])  # [128]


def verify(claim, z_skip, x_outer, s_hat_v, ch: Sha256Challenger):
    """Observe LABEL + s_hat_v, check s_hat_v encodes `claim` at (z_skip, x_outer[0]),
    sample r'', reduce to the BaseFold sumcheck claim. Returns
    (sumcheck_claim, eq_r_dprime, ok)."""
    ch._t = fs.observe_label(ch._t, LABEL)
    ch._t = fs.observe_slice(ch._t, s_hat_v)
    ok = _inner_product(_build_claim_weights(z_skip, x_outer[0]), s_hat_v) == claim
    ch._t, r_dprime = fs.sample_slice(ch._t, LOG_PACKING)
    eq_r_dprime = sumcheck.build_eq(r_dprime)
    sumcheck_claim = _inner_product(tensor_algebra_transpose(s_hat_v), eq_r_dprime)
    return sumcheck_claim, eq_r_dprime, ok
