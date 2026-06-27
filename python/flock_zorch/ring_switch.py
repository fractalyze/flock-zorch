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
import jax.numpy as jnp

from flock_zorch import field, sumcheck
from flock_zorch.challenger import Challenger

LOG_PACKING = 7
LABEL = b"flock-ring-switch-v0"
_W64 = (np.uint64(1) << np.arange(64, dtype=np.uint64))


def _to_bits(arr: np.ndarray) -> np.ndarray:
    """uint64 [n, 2] F128 -> uint8 [n, 128]: bit (r<64 from lo, else hi)."""
    lo = ((arr[:, 0:1] >> np.arange(64, dtype=np.uint64)) & 1).astype(np.uint8)
    hi = ((arr[:, 1:2] >> np.arange(64, dtype=np.uint64)) & 1).astype(np.uint8)
    return np.concatenate([lo, hi], axis=1)   # [n, 128]


def _from_bits(bits: np.ndarray) -> np.ndarray:
    """uint8 [n, 128] -> uint64 [n, 2] F128 (bit r at position r)."""
    b = bits.astype(np.uint64)
    lo = (b[:, :64] * _W64).sum(axis=1, dtype=np.uint64)
    hi = (b[:, 64:] * _W64).sum(axis=1, dtype=np.uint64)
    return np.stack([lo, hi], axis=1)


def fold_1b_rows(packed_witness: np.ndarray, suffix_tensor, mul=field.mul) -> np.ndarray:
    """s_hat_v[r] = Σ_i bit_r(witness[i]) · suffix_tensor[i]. Returns uint64 [128, 2]."""
    bits = jnp.asarray(_to_bits(np.asarray(packed_witness)).astype(np.uint64))  # [n,128]
    st = jnp.asarray(suffix_tensor)                                             # [n,2]
    sel = bits[:, :, None] * st[:, None, :]    # bit_r·suffix[i]  [n,128,2]
    return np.asarray(sumcheck._xor_reduce(sel, axis=0))                        # [128,2]


def fold_b128_elems(suffix_tensor, eq_r_dprime, mul=field.mul) -> np.ndarray:
    """rs_eq_ind[i] = Σ_b bit_b(suffix_tensor[i]) · eq_r_dprime[b]. uint64 [n, 2]."""
    bits = jnp.asarray(_to_bits(np.asarray(suffix_tensor)).astype(np.uint64))   # [n,128]
    eq = jnp.asarray(eq_r_dprime)                                               # [128,2]
    sel = bits[:, :, None] * eq[None, :, :]    # [n,128,2]
    return np.asarray(sumcheck._xor_reduce(sel, axis=1))                        # [n,2]


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
    suffix_tensor = sumcheck.build_eq(suffix, mul=mul)         # [2^L, 2]
    s_hat_v = fold_1b_rows(packed_witness, suffix_tensor, mul)  # [128,2]
    ch.observe_f128_slice(s_hat_v)
    r_dprime = jnp.asarray(ch.sample_f128_vec(LOG_PACKING))    # [7,2]
    eq_r_dprime = sumcheck.build_eq(r_dprime, mul=mul)         # [128,2]
    s_hat_u = tensor_algebra_transpose(s_hat_v)
    sumcheck_claim = inner_product(s_hat_u, eq_r_dprime, mul)  # [2]
    rs_eq_ind = fold_b128_elems(suffix_tensor, eq_r_dprime, mul)  # [2^L,2]
    return s_hat_v, rs_eq_ind, sumcheck_claim
