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
from frx import lax

from flock_zorch import fs, ghash, sumcheck
from flock_zorch.challenger import Challenger

LOG_PACKING = ghash.LOG_PACKING
LABEL = b"flock-ring-switch-v0"


def _zorch_ring_switch():
    # zorch's bounded kernels import Pallas. Keep that import on the GPU path so
    # hermetic CPU tests do not need to initialize a Mosaic runtime.
    from zorch.pcs import ring_switch

    return ring_switch


def _bits(x):
    limbs = lax.bitcast_convert_type(x, fnp.uint32)
    shifts = fnp.arange(32, dtype=fnp.uint32)
    return ((limbs[..., None] >> shifts) & fnp.uint32(1)).reshape(*x.shape, -1)


def _from_bits(bits, dtype):
    weights = fnp.uint32(1) << fnp.arange(32, dtype=fnp.uint32)
    limbs = fnp.sum(
        bits.reshape(*bits.shape[:-1], -1, 32) * weights,
        axis=-1,
        dtype=fnp.uint32,
    )
    return lax.bitcast_convert_type(limbs, dtype)


def _bit_slice_evals(packed, tensor):
    if frx.default_backend() == "gpu":
        return _zorch_ring_switch().bit_slice_evals(packed, tensor)
    selected = lax.bitcast_convert_type(tensor, fnp.uint32)[..., None, :] * _bits(
        packed
    )[..., None]
    return fnp.sum(lax.bitcast_convert_type(selected, tensor.dtype), axis=-2)


def _rs_eq_ind(tensor, eq_r_dprime):
    if frx.default_backend() == "gpu":
        return _zorch_ring_switch().rs_eq_ind(tensor, eq_r_dprime)
    selected = _bits(tensor)[..., None] * lax.bitcast_convert_type(
        eq_r_dprime, fnp.uint32
    )[None, ...]
    return fnp.sum(lax.bitcast_convert_type(selected, eq_r_dprime.dtype), axis=-1)


def _tensor_algebra_transpose(v):
    return _from_bits(_bits(v).swapaxes(-1, -2), v.dtype)


def _inner_product(a, b):
    return fnp.sum(a * b, axis=0)


@frx.jit
def _reduce_one(t, packed, x_outer):
    """One claim's observe-and-reduce (prove_batched runs it per opening point):
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
    s_hat_v = _bit_slice_evals(packed, suffix_tensor)     # (128,) ghash
    t = fs.observe_slice(t, s_hat_v)                  # observe device ghash directly
    t, r_dprime = fs.sample_slice(t, LOG_PACKING)     # [7] ghash, kept native
    eq_r_dprime = sumcheck.build_eq(r_dprime)          # [128] ghash, for the gamma combine
    claim = _inner_product(_tensor_algebra_transpose(s_hat_v), eq_r_dprime)
    return t, s_hat_v, suffix_tensor, eq_r_dprime, claim       # claim native ghash


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
        rs_eq_inds.append(_rs_eq_ind(suffix_tensor, scaled))  # ghash [2^L], device-resident
        s_hat_vs.append(s_hat_v)
        sumcheck_claims.append(claim)
    return s_hat_vs, rs_eq_inds, sumcheck_claims, gammas


# ---- verifier side ---------------------------------------------------------

from flock_zorch.zerocheck import _lagrange_weights  # noqa: E402

_ONE_G = ghash.to_ghash(fnp.array([1, 0], fnp.uint64))
_CLAIM_K = LOG_PACKING - 1  # 6: the φ8 skip dim; bit-6 carries the x_outer[0] eq split


def _build_claim_weights(z_skip, x_outer_0):
    """flock `ring_switch::build_claim_weights`: lam(z_skip) ⊗ eq(x_outer[0]).
    i∈[0,64) take eq(x0,0)=1+x0; i∈[64,128) take eq(x0,1)=x0."""
    lam = _lagrange_weights(_CLAIM_K, z_skip, 0)  # [64] ghash, φ8 S-domain
    return fnp.concatenate([lam * (_ONE_G + x_outer_0), lam * x_outer_0])  # [128]


def verify(claim, z_skip, x_outer, s_hat_v, ch: Challenger):
    """Observe LABEL + s_hat_v, check s_hat_v encodes `claim` at (z_skip, x_outer[0]),
    sample r'', reduce to the BaseFold sumcheck claim. Returns
    (sumcheck_claim, eq_r_dprime, ok)."""
    s_hat_v = ghash.to_ghash(fnp.asarray(ghash.to_lanes(s_hat_v)))  # native or lanes → native
    ch._t = fs.observe_label(ch._t, LABEL)
    ch._t = fs.observe_slice(ch._t, s_hat_v)
    ok = _inner_product(_build_claim_weights(z_skip, x_outer[0]), s_hat_v) == claim
    ch._t, r_dprime = fs.sample_slice(ch._t, LOG_PACKING)
    eq_r_dprime = sumcheck.build_eq(r_dprime)
    sumcheck_claim = _inner_product(_tensor_algebra_transpose(s_hat_v), eq_r_dprime)
    return sumcheck_claim, eq_r_dprime, ok
