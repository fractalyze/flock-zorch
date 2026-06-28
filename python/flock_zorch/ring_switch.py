"""flock's ring-switching reduction (`pcs::ring_switch::prove`), authored in jax —
byte-identical to flock-core. Bridges a bit-MLE claim to the packed-MLE BaseFold
open: computes `s_hat_v` (the 128 bit-slice partial evals, the only field on the
wire), the BaseFold target `sumcheck_claim`, and the transparent multilinear
`b = rs_eq_ind` that BaseFold runs its sumcheck against.

All ops are binary-bit × F128 reductions (vectorized) + a 128×128 bit-transpose:
  suffix_tensor = eq(x_outer[1:])
  s_hat_v[r]    = Σ_i bit_r(witness[i]) · suffix_tensor[i]          (fold_1b_rows)
  sumcheck_claim= ⟨transpose(s_hat_v), eq(r_dprime)⟩
  rs_eq_ind[i]  = Σ_b bit_b(suffix_tensor[i]) · eq(r_dprime)[b]     (fold_b128_elems)

Challenger: observe `flock-ring-switch-v0`, observe_f128_slice(s_hat_v), sample
r_dprime[7]. Requires `jax_enable_x64`.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from flock_zorch import field, sumcheck
from flock_zorch.challenger import Challenger

U64 = jnp.uint64
LOG_PACKING = 7
LABEL = b"flock-ring-switch-v0"
_W64 = (np.uint64(1) << np.arange(64, dtype=np.uint64))


def _to_bits_dev(arr):
    """jnp uint64 [n, 2] F128 -> uint64 [n, 128]: bit r (r<64 from lo, else hi). Device."""
    r = jnp.arange(64, dtype=U64)
    lo = (arr[:, 0:1] >> r) & U64(1)
    hi = (arr[:, 1:2] >> r) & U64(1)
    return jnp.concatenate([lo, hi], axis=1)   # [n, 128]


def _to_bits(arr: np.ndarray) -> np.ndarray:
    """uint64 [n, 2] -> uint8 [n, 128] (host; only for the tiny 128×128 transpose)."""
    lo = ((arr[:, 0:1] >> np.arange(64, dtype=np.uint64)) & 1).astype(np.uint8)
    hi = ((arr[:, 1:2] >> np.arange(64, dtype=np.uint64)) & 1).astype(np.uint8)
    return np.concatenate([lo, hi], axis=1)


def _from_bits(bits: np.ndarray) -> np.ndarray:
    """uint8 [n, 128] -> uint64 [n, 2] F128 (bit r at position r) — host (small, 128×)."""
    b = bits.astype(np.uint64)
    lo = (b[:, :64] * _W64).sum(axis=1, dtype=np.uint64)
    hi = (b[:, 64:] * _W64).sum(axis=1, dtype=np.uint64)
    return np.stack([lo, hi], axis=1)


@jax.jit
def _fold_1b_rows_dev(witness, suffix):
    """s_hat_v[r] = Σ_i bit_r(witness[i])·suffix[i]. witness/suffix [n,2] -> [128,2].
    Device + jit so the large [n,128,2] intermediate stays fused on device, off HBM."""
    bits = _to_bits_dev(witness)                       # [n,128]
    return sumcheck._xor_reduce(bits[:, :, None] * suffix[:, None, :], axis=0)


@jax.jit
def _fold_b128_elems_dev(suffix, eq):
    """rs_eq_ind[i] = Σ_b bit_b(suffix[i])·eq[b]. suffix [n,2], eq [128,2] -> [n,2]."""
    bits = _to_bits_dev(suffix)                        # [n,128]
    return sumcheck._xor_reduce(bits[:, :, None] * eq[None, :, :], axis=1)


def fold_1b_rows(packed_witness, suffix_tensor, mul=field.mul) -> np.ndarray:
    return np.asarray(_fold_1b_rows_dev(jnp.asarray(packed_witness), jnp.asarray(suffix_tensor)))


def fold_b128_elems(suffix_tensor, eq_r_dprime, mul=field.mul) -> np.ndarray:
    return np.asarray(_fold_b128_elems_dev(jnp.asarray(suffix_tensor), jnp.asarray(eq_r_dprime)))


def tensor_algebra_transpose(s_hat_v) -> np.ndarray:
    """128×128 bit transpose: s_hat_u[b].bit_iskip = s_hat_v[iskip].bit_b."""
    bits = _to_bits(np.asarray(s_hat_v))       # [128,128] bits[i_skip, b]
    return _from_bits(bits.T)                  # s_hat_u[b] from column b


def inner_product(a, b, mul=field.mul) -> np.ndarray:
    return np.asarray(sumcheck._xor_reduce(mul(jnp.asarray(a), jnp.asarray(b)), axis=0))


def prove(packed_witness, x_outer, ch: Challenger, mul=field.mul):
    """Returns (s_hat_v [128,2], rs_eq_ind [2^L,2], sumcheck_claim [2]).
    Byte-identical to flock `ring_switch::prove`."""
    ch.observe_label(LABEL)
    suffix = jnp.asarray(np.asarray(x_outer)[1:])              # x_outer[1:], length L
    suffix_tensor = sumcheck.build_eq_fused(suffix, mul=mul)         # [2^L, 2]
    s_hat_v = fold_1b_rows(packed_witness, suffix_tensor, mul)  # [128,2]
    ch.observe_f128_slice(s_hat_v)
    r_dprime = jnp.asarray(ch.sample_f128_vec(LOG_PACKING))    # [7,2]
    eq_r_dprime = sumcheck.build_eq_fused(r_dprime, mul=mul)         # [128,2]
    s_hat_u = tensor_algebra_transpose(s_hat_v)
    sumcheck_claim = inner_product(s_hat_u, eq_r_dprime, mul)  # [2]
    rs_eq_ind = fold_b128_elems(suffix_tensor, eq_r_dprime, mul)  # [2^L,2]
    return s_hat_v, rs_eq_ind, sumcheck_claim


def prove_batched(packed_witness, x_outers, ch: Challenger, mul=field.mul):
    """Batched ring-switch over N opening points — byte-identical to flock
    `ring_switch::prove_batched_padded_with_precomputed` (dense; no precompute —
    each s_hat_v value equals `fold_1b_rows`, so the observed bytes match).

    Transcript: per claim (in order) observe `flock-ring-switch-v0` + s_hat_v +
    sample r_dprime[7]; THEN sample N γ's (Schwartz-Zippel-sound, after all
    observations); THEN bake γ_i into each `rs_eq_ind_i`. Returns
    (s_hat_vs, rs_eq_inds[γ-baked], sumcheck_claims, gammas)."""
    works = []
    for x_outer in x_outers:
        ch.observe_label(LABEL)
        suffix = jnp.asarray(np.asarray(x_outer)[1:])
        suffix_tensor = sumcheck.build_eq_fused(suffix, mul=mul)
        s_hat_v = fold_1b_rows(packed_witness, suffix_tensor, mul)
        ch.observe_f128_slice(s_hat_v)
        r_dprime = jnp.asarray(ch.sample_f128_vec(LOG_PACKING))
        eq_r_dprime = sumcheck.build_eq_fused(r_dprime, mul=mul)
        s_hat_u = tensor_algebra_transpose(s_hat_v)
        sumcheck_claim = inner_product(s_hat_u, eq_r_dprime, mul)
        works.append((s_hat_v, suffix_tensor, eq_r_dprime, sumcheck_claim))

    gammas = [ch.sample_f128() for _ in range(len(x_outers))]
    s_hat_vs, rs_eq_inds, sumcheck_claims = [], [], []
    for (s_hat_v, suffix_tensor, eq_r_dprime, sc), g in zip(works, gammas):
        scaled = mul(jnp.asarray(g), jnp.asarray(eq_r_dprime))   # γ baked into eq
        rs_eq_inds.append(fold_b128_elems(suffix_tensor, scaled, mul))
        s_hat_vs.append(s_hat_v)
        sumcheck_claims.append(sc)
    return s_hat_vs, rs_eq_inds, sumcheck_claims, gammas
