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

import functools

import numpy as np
import jax
import jax.numpy as jnp

from flock_zorch import field
from flock_zorch.sumcheck import build_eq_fused, ONE
from flock_zorch.zerocheck import _lagrange_weights
from flock_zorch.field import _to_int, _to_lohi
from flock_zorch.challenger import Challenger
from flock_zorch._csc_fold import _flatten_nz, _csc_segments, _seg_xor_fold

U64 = jnp.uint64
LABEL = b"flock-lincheck-v0"


def build_quirky_eq_table(z_skip_int: int, x_inner_rest, k_skip: int, mul=field.mul):
    """eq_inner[i_skip + i_rest·2^k_skip] = λ_skip[i_skip]·eq_rest[i_rest]
    (flock `build_quirky_eq_table`; i_skip in the LOW bits)."""
    lam = _lagrange_weights(k_skip, z_skip_int, 0)             # S-domain, len 2^k_skip
    lam = jnp.asarray(np.stack([_to_lohi(x) for x in lam]))    # [ell_skip, 2]
    eq_rest = build_eq_fused(jnp.asarray(x_inner_rest), mul=mul)  # [ell_rest, 2] — fused (avoids per-layer eager dispatch)
    prod = mul(eq_rest[:, None, :], lam[None, :, :])           # [ell_rest, ell_skip, 2]
    return prod.reshape(-1, 2)


def _mat_fold(mat_dense, eq, mul=field.mul):
    """Transposed binary-matrix·vector: out[c] = Σ_{r: M[r,c]=1} eq[r].

    mat_dense: uint64 [k, k] (0/1, indexed [row, col]); eq: [k, 2] -> [k, 2]."""
    sel = mat_dense[:, :, None] * eq[:, None, :]              # M[r,c]·eq[r]  (select)
    return field.sum(sel, axis=0)                           # XOR over rows -> [c, 2]


def fold_alpha_batched(alpha, a_dense, b_dense, eq_inner, mul=field.mul):
    """comb[c] = α·(A₀ᵀ·eq_inner)[c] ⊕ (B₀ᵀ·eq_inner)[c] (flock
    `sparse_row_fold_alpha_batched`)."""
    ae = _mat_fold(a_dense, eq_inner, mul)
    be = _mat_fold(b_dense, eq_inner, mul)
    return field.add(mul(alpha, ae), be)


class CscCircuit:
    """Sparse lincheck circuit (flock `CscCircuit`) for real hash R1CS where k =
    2^k_log is too large for dense [k,k] matrices (sha2 k=32768, blake3 k=16384).
    Holds A₀/B₀ as flat nonzero (col,row) pairs; `fold_alpha_batched` is the
    transposed binary matvec out[c] = α·Σ_{r:A[r,c]=1} eq[r] ⊕ Σ_{r:B[r,c]=1} eq[r].
    Runs **on device** as a column-sorted prefix-XOR scan (`_seg_xor_fold`) — handles
    the skewed const_pin column degree (a padded gather would blow up, an atomic
    XOR-scatter would hotspot) without either. `const_pin` carries the +β pin column.
    (The construction-time column sort is host, once.)"""

    def __init__(self, a0_rows, b0_rows, k: int, const_pin=None):
        self.k = k
        self.const_pin = const_pin
        a_col, a_row = _flatten_nz(a0_rows)
        b_col, b_row = _flatten_nz(b0_rows)
        self._a_seg = _csc_segments(a_col, a_row)
        self._b_seg = _csc_segments(b_col, b_row)

    def fold_alpha_batched(self, alpha, eq_inner, mul=field.mul):
        eq = jnp.asarray(np.asarray(eq_inner, np.uint64).reshape(-1, 2))
        zero = jnp.zeros((self.k, 2), U64)
        out_a = _seg_xor_fold(eq, *self._a_seg, self.k) if self._a_seg else zero
        out_b = _seg_xor_fold(eq, *self._b_seg, self.k) if self._b_seg else zero
        return field.add(mul(jnp.asarray(alpha), out_a), out_b)


def partial_fold_packed_z(z_packed_bytes: bytes, m: int, k_log: int, eq_outer, mul=field.mul):
    """z_vec[i_inner] = Σ_{i_outer} z(i_inner, i_outer)·eq_outer[i_outer]
    (flock `partial_fold_packed_z`, useful_bits = 2^k_log).

    z_packed layout: byte `z_packed[byte_idx·k + i_inner]` holds outer bits
    `z[i_inner, 8·byte_idx + r]` at bit r."""
    k = 1 << k_log
    n_outer = 1 << (m - k_log)
    n_bytes = n_outer // 8
    zp = jnp.asarray(np.frombuffer(z_packed_bytes, np.uint8).reshape(n_bytes, k))
    return _partial_fold_dev(zp, eq_outer, n_outer)         # device + jit (keeps the intermediate off HBM)


@functools.partial(jax.jit, static_argnums=(2,))
def _partial_fold_dev(zp, eq_outer, n_outer):
    """z_vec[i_inner] = Σ_{i_outer} bit·eq_outer[i_outer], device+jit so the large
    [n_outer,k,2] intermediate stays fused on device and never lands in HBM."""
    bits = ((zp[:, None, :] >> jnp.arange(8, dtype=jnp.uint8)[None, :, None]) & 1)  # [nb,8,k]
    bits = bits.reshape(n_outer, zp.shape[1]).astype(jnp.uint64)                    # i_outer=byte·8+r
    return field.sum(bits[:, :, None] * eq_outer[:, None, :], axis=0)             # [k, 2]


def _round_eval(c, z, mul=field.mul):
    """Product-sumcheck round message (q(1), q(∞)) over the TOP-bit split (flock
    `sumcheck_round_eval`): half = len/2; (Σ chi·zhi, Σ (chi+clo)(zhi+zlo))."""
    half = c.shape[0] // 2
    clo, chi = c[:half], c[half:]
    zlo, zhi = z[:half], z[half:]
    e1 = field.sum(mul(chi, zhi))
    einf = field.sum(mul(field.add(chi, clo), field.add(zhi, zlo)))
    return e1, einf


def _bind_top(v, r, mul=field.mul):
    """Bind the top variable at r (flock `sumcheck_bind_top`):
    v'[i] = v[i] + r·(v[i+half] + v[i]); length halves."""
    half = v.shape[0] // 2
    vlo, vhi = v[:half], v[half:]
    return field.add(vlo, mul(r, field.add(vhi, vlo)))


def prove(z_packed_bytes, a_dense, b_dense, x_ab, m, k_log, k_skip,
          domain=b"flock-test-v0", mul=field.mul, ch=None, capture=False, circuit=None):
    """Run lincheck. x_ab = dict(z_skip:[2], x_inner_rest:[*,2], x_outer:[*,2]).
    Byte-identical to flock `lincheck::prove`/`prove_padded_capture_z_vec`.

    `circuit`: a `CscCircuit` for real hash R1CS (sparse A₀/B₀ at large k, with an
    optional const_pin +β column); when None, the dense `a_dense`/`b_dense` path is
    used (small test R1CS). Default returns (rounds, z_partial). With `capture=True`
    (the e2e fused prover) also returns the post-sumcheck claim and the pre-sumcheck
    z_vec. Pass a shared `ch` to thread Fiat-Shamir; else a fresh Challenger(domain)."""
    inner_rest = k_log - k_skip
    if ch is None:
        ch = Challenger(domain)
    ch.observe_label(LABEL)
    alpha = jnp.asarray(ch.sample_f128())

    eq_inner = build_quirky_eq_table(_to_int(x_ab["z_skip"]), x_ab["x_inner_rest"], k_skip, mul)
    if circuit is not None:
        comb = jnp.asarray(circuit.fold_alpha_batched(alpha, eq_inner, mul))
        if circuit.const_pin is not None:
            beta = jnp.asarray(ch.sample_f128())          # sampled AFTER alpha (flock order)
            col = circuit.const_pin
            comb = comb.at[col].set(field.add(comb[col], beta))
    else:
        comb = fold_alpha_batched(alpha, jnp.asarray(a_dense), jnp.asarray(b_dense), eq_inner, mul)

    eq_outer = build_eq_fused(jnp.asarray(x_ab["x_outer"]), mul=mul)
    z_vec = partial_fold_packed_z(z_packed_bytes, m, k_log, eq_outer, mul)
    z_vec_pre = np.asarray(z_vec) if capture else None  # pre-sumcheck (PCS open reuse)

    # Unfused on purpose: each round is _round_eval then _bind_top, mirroring flock's
    # steps. Do NOT hand-fuse into Rust's sumcheck_bind_both_and_eval_next — operator
    # fusion is the zkx compiler's job (see CLAUDE.md).
    rounds, r_rounds = [], []
    if inner_rest > 0:
        e1, einf = _round_eval(comb, z_vec, mul)
        for t in range(inner_rest):
            ch.observe_f128(e1)
            ch.observe_f128(einf)
            r = jnp.asarray(ch.sample_f128())
            rounds.append((np.asarray(e1), np.asarray(einf)))
            r_rounds.append(r)
            comb = _bind_top(comb, r, mul)
            z_vec = _bind_top(z_vec, r, mul)
            if t + 1 < inner_rest:
                e1, einf = _round_eval(comb, z_vec, mul)
    z_partial = np.asarray(z_vec)
    if not capture:
        return rounds, z_partial

    # ---- claim derivation (flock prove_padded_inner steps 6-9) ----
    ch.observe_f128_slice(z_partial)                      # 6. observe z_partial
    r_inner_skip = ch.sample_f128()                       # 7. fresh z_skip AFTER
    lam = _lagrange_weights(k_skip, _to_int(r_inner_skip), 0)  # 8. φ8 S-domain weights
    lam_arr = jnp.asarray(np.stack([_to_lohi(x) for x in lam]))
    w = np.asarray(field.sum(mul(lam_arr, jnp.asarray(z_partial)), axis=0))  # inner_product
    r_inner_rest = [np.asarray(r) for r in reversed(r_rounds)]  # 9. LSB-first
    claim = {"r_inner_skip": np.asarray(r_inner_skip),
             "r_inner_rest": np.stack(r_inner_rest) if r_inner_rest else np.zeros((0, 2), np.uint64),
             "w": w}
    return rounds, z_partial, claim, z_vec_pre
