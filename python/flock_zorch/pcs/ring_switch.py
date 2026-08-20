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
def s_hat_v_from_z_vec(z_vec, tail):
    """The ab claim's `s_hat_v` from the lincheck's initial z_vec — flock
    `ring_switch::s_hat_v_from_z_vec`: `s_hat_v[b] = Σ_k eq(tail)[k]·z_vec[k·128+b]`.

    This is the authority for why the two derivations agree: `z_vec`
    (`[2^k_log]` ghash, `partial_fold_packed_z`'s output) already folded the
    witness over `x_outer`, and `eq(x_full[1:])` factorizes as
    `eq(tail) ⊗ eq(x_outer)`, so folding the KiB-scale z_vec by `eq(tail)`
    reassociates the exact GF(2¹²⁸) sum `bit_slice_evals` computes from the
    full packed witness — bit-identical values, no wire change. `tail` is
    `r_inner_rest[1:]`: the LOG_PACKING boundary ate coordinate 0
    (k_skip + 1 = LOG_PACKING), and an empty tail (k_log == LOG_PACKING)
    degenerates to `eq([]) = [1]`, i.e. z_vec IS s_hat_v. Staged here rather
    than in `zorch.pcs.ring_switch` to avoid a zorch+frx lockstep bump;
    promote alongside the next pin bump."""
    eq_tail = sumcheck.build_eq(tail)  # (2^tail,) ghash; [1] when tail is empty
    return fnp.sum(eq_tail[:, None] * z_vec.reshape(-1, 1 << LOG_PACKING), axis=0)


@frx.jit
def _slice_evals(packed, suffixes, precomputed):
    """The transcript-independent half of the batched open, one read per claim
    that lacks a precomputed `s_hat_v`.

    Builds each opening point's suffix eq tensor (always — `rs_eq_ind` consumes
    it) and bit-slices the witness only for claims whose `precomputed` entry is
    None; a `[128]` ghash entry (see `s_hat_v_from_z_vec`) short-circuits that
    claim's witness read. Per-claim beats the
    shared batched read on both of its costs: the single-claim elements kernel
    takes the two-row tile the batched kernel cannot (its batch axis already
    owns the registers), and the batched form additionally materializes the
    `(2^L, N)` suffix stack — glue that costs more at phase level than the
    shared witness read saves. `suffixes` is a tuple of N `[L]` ghash coord
    vectors (static N); the build stays under this jit so `build_eq` fuses.
    Returns (s_hat_vs: N x `[128]` ghash, suffix_tensors: N contiguous `[2^L]`
    ghash — `rs_eq_ind` reads each contiguously)."""
    suffix_tensors = [sumcheck.build_eq(s) for s in suffixes]  # N x (2^L,) ghash
    s_hat_vs = [
        p if p is not None else bit_slice_evals(packed, t)
        for p, t in zip(precomputed, suffix_tensors)
    ]
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


def prove_batched(
    packed_witness, x_outers, ch: Sha256Challenger, precomputed_s_hat_vs=None
):
    """Batched ring-switch over N opening points — byte-identical to flock
    `ring_switch::prove_batched_padded_with_precomputed`.

    The transcript-free witness reduction (`_slice_evals`) is lifted out of the
    serial per-claim Fiat-Shamir (`_observe_and_reduce`), which runs in the same
    order as before, so the wire is unchanged. `precomputed_s_hat_vs` (optional,
    length N, entries None or `[128]` ghash) supplies claims whose s_hat_v was
    already derived (see `s_hat_v_from_z_vec` — same values, same wire).

    Transcript: per claim (in order) observe `flock-ring-switch-v0` + s_hat_v +
    sample r_dprime[7]; THEN sample N gamma's (sound only after all observations);
    THEN bake gamma_i into each `rs_eq_ind_i` (the caller-owned linear combination
    — see the zorch module's contract). Returns
    (s_hat_vs, rs_eq_inds[gamma-baked], sumcheck_claims, gammas)."""
    packed = ghash.to_ghash(packed_witness)
    suffixes = tuple(x_outer[1:] for x_outer in x_outers)  # ghash coords, length L
    precomputed = precomputed_s_hat_vs or (None,) * len(x_outers)
    if len(precomputed) != len(x_outers):
        raise ValueError(
            f"precomputed_s_hat_vs: expected {len(x_outers)} entries, "
            f"got {len(precomputed)}"
        )
    for p in precomputed:
        if p is not None and p.shape != (1 << LOG_PACKING,):
            raise ValueError(f"precomputed s_hat_v must be [128] ghash, got {p.shape}")
    s_hat_vs, suffix_tensors = _slice_evals(packed, suffixes, precomputed)

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
