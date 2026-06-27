"""flock's lincheck `prove` — the second PIOP sub-protocol — authored as a host
round loop, byte-identical to flock-core's `lincheck::prove_padded_inner`
(`SparseMatrixCircuit`, no const-pin).

Reduces the zerocheck's â/b̂ claims to one z-claim. Structure (flock §protocol):
  observe `flock-lincheck-v0` → sample α → comb_vec = α·(A₀ᵀ·eq_inner) ⊕ (B₀ᵀ·eq_inner)
  → z_vec = partial-fold of z at x_outer → (k_log−k_skip)-round product sumcheck
  → send z_partial (length 2^k_skip). Proof = {rounds:(e1,einf)…, z_partial}.

Two conventions DIFFER from zerocheck's multilinear rounds (so this has its own
round/bind): lincheck's sumcheck binds the **top** bit (split at half, not the
interleaved (2x,2x+1) pairs) and carries **no eq factor** (plain product sum
`Σ comb·z`). The eq tables (`build_eq`) and φ8 Lagrange weights are shared.

Reuses zorch via the challenger (`zorch.byte_transcript`). Requires
`jax_enable_x64` and `zorch` on PYTHONPATH.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from flock_zorch import field
from flock_zorch.sumcheck import build_eq, _xor_reduce, ONE
from flock_zorch.zerocheck import _lagrange_weights, _to_int, _to_lohi
from flock_zorch.challenger import Challenger

U64 = jnp.uint64
LABEL = b"flock-lincheck-v0"


def build_quirky_eq_table(z_skip_int: int, x_inner_rest, k_skip: int, mul=field.mul):
    """eq_inner[i_skip + i_rest·2^k_skip] = λ_skip[i_skip]·eq_rest[i_rest]
    (flock `build_quirky_eq_table`; i_skip in the LOW bits)."""
    lam = _lagrange_weights(k_skip, z_skip_int, 0)             # S-domain, len 2^k_skip
    lam = jnp.asarray(np.stack([_to_lohi(x) for x in lam]))    # [ell_skip, 2]
    eq_rest = build_eq(jnp.asarray(x_inner_rest), mul=mul)     # [ell_rest, 2]
    prod = mul(eq_rest[:, None, :], lam[None, :, :])           # [ell_rest, ell_skip, 2]
    return prod.reshape(-1, 2)


def _mat_fold(mat_dense, eq, mul=field.mul):
    """Transposed binary-matrix·vector: out[c] = Σ_{r: M[r,c]=1} eq[r].

    mat_dense: uint64 [k, k] (0/1, indexed [row, col]); eq: [k, 2] -> [k, 2]."""
    sel = mat_dense[:, :, None] * eq[:, None, :]              # M[r,c]·eq[r]  (select)
    return _xor_reduce(sel, axis=0)                           # XOR over rows -> [c, 2]


def fold_alpha_batched(alpha, a_dense, b_dense, eq_inner, mul=field.mul):
    """comb[c] = α·(A₀ᵀ·eq_inner)[c] ⊕ (B₀ᵀ·eq_inner)[c] (flock
    `sparse_row_fold_alpha_batched`)."""
    ae = _mat_fold(a_dense, eq_inner, mul)
    be = _mat_fold(b_dense, eq_inner, mul)
    return field.add(mul(alpha, ae), be)


def partial_fold_packed_z(z_packed_bytes: bytes, m: int, k_log: int, eq_outer, mul=field.mul):
    """z_vec[i_inner] = Σ_{i_outer} z(i_inner, i_outer)·eq_outer[i_outer]
    (flock `partial_fold_packed_z`, useful_bits = 2^k_log).

    z_packed layout: byte `z_packed[byte_idx·k + i_inner]` holds outer bits
    `z[i_inner, 8·byte_idx + r]` at bit r."""
    k = 1 << k_log
    n_outer = 1 << (m - k_log)
    n_bytes = n_outer // 8
    zp = np.frombuffer(z_packed_bytes, np.uint8).reshape(n_bytes, k)
    bits = (zp[:, None, :] >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1  # [nb,8,k]
    bits = jnp.asarray(bits.reshape(n_outer, k).astype(np.uint64))             # i_outer=byte·8+r
    sel = bits[:, :, None] * eq_outer[:, None, :]            # bit·eq_outer[i_outer]
    return _xor_reduce(sel, axis=0)                          # XOR over i_outer -> [k, 2]


def _round_eval(c, z, mul=field.mul):
    """Product-sumcheck round message (q(1), q(∞)) over the TOP-bit split (flock
    `sumcheck_round_eval`): half = len/2; (Σ chi·zhi, Σ (chi+clo)(zhi+zlo))."""
    half = c.shape[0] // 2
    clo, chi = c[:half], c[half:]
    zlo, zhi = z[:half], z[half:]
    e1 = _xor_reduce(mul(chi, zhi))
    einf = _xor_reduce(mul(field.add(chi, clo), field.add(zhi, zlo)))
    return e1, einf


def _bind_top(v, r, mul=field.mul):
    """Bind the top variable at r (flock `sumcheck_bind_top`):
    v'[i] = v[i] + r·(v[i+half] + v[i]); length halves."""
    half = v.shape[0] // 2
    vlo, vhi = v[:half], v[half:]
    return field.add(vlo, mul(r, field.add(vhi, vlo)))


def prove(z_packed_bytes, a_dense, b_dense, x_ab, m, k_log, k_skip,
          domain=b"flock-test-v0", mul=field.mul):
    """Run lincheck. x_ab = dict(z_skip:[2], x_inner_rest:[*,2], x_outer:[*,2]).
    Returns (rounds: list[(e1,einf)], z_partial: [2^k_skip, 2]). Byte-identical to
    flock `lincheck::prove` (SparseMatrixCircuit, no const-pin)."""
    inner_rest = k_log - k_skip
    ch = Challenger(domain)
    ch.observe_label(LABEL)
    alpha = jnp.asarray(ch.sample_f128())

    eq_inner = build_quirky_eq_table(_to_int(x_ab["z_skip"]), x_ab["x_inner_rest"], k_skip, mul)
    comb = fold_alpha_batched(alpha, jnp.asarray(a_dense), jnp.asarray(b_dense), eq_inner, mul)
    # SparseMatrixCircuit::new has const_pin = None -> no β step.

    eq_outer = build_eq(jnp.asarray(x_ab["x_outer"]), mul=mul)
    z_vec = partial_fold_packed_z(z_packed_bytes, m, k_log, eq_outer, mul)

    rounds = []
    if inner_rest > 0:
        e1, einf = _round_eval(comb, z_vec, mul)
        for t in range(inner_rest):
            ch.observe_f128(e1)
            ch.observe_f128(einf)
            r = jnp.asarray(ch.sample_f128())
            rounds.append((np.asarray(e1), np.asarray(einf)))
            comb = _bind_top(comb, r, mul)
            z_vec = _bind_top(z_vec, r, mul)
            if t + 1 < inner_rest:
                e1, einf = _round_eval(comb, z_vec, mul)
    return rounds, np.asarray(z_vec)
