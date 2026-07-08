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
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np
import jax
import jax.numpy as jnp

from flock_zorch import field
from flock_zorch.sumcheck import build_eq_fused, ONE
from flock_zorch.zerocheck import _lagrange_weights
from flock_zorch.field import _to_int, _to_lohi
from flock_zorch.challenger import Challenger
from flock_zorch._csc_fold import _flatten_nz, _csc_segments, _seg_xor_fold
from zorch.round import ProveChain, Round

U64 = jnp.uint64
LABEL = b"flock-lincheck-v0"


def build_quirky_eq_table(z_skip_int: int, x_inner_rest, k_skip: int):
    """eq_inner[i_skip + i_rest·2^k_skip] = λ_skip[i_skip]·eq_rest[i_rest]
    (flock `build_quirky_eq_table`; i_skip in the LOW bits)."""
    lam = _lagrange_weights(k_skip, z_skip_int, 0)             # S-domain, len 2^k_skip
    lam = field.to_ghash(jnp.asarray(np.stack([_to_lohi(x) for x in lam])))  # [ell_skip]
    eq_rest = field.to_ghash(build_eq_fused(jnp.asarray(x_inner_rest)))  # [ell_rest] — fused (avoids per-layer eager dispatch)
    prod = eq_rest[:, None] * lam[None, :]                    # [ell_rest, ell_skip]
    return field.from_ghash(prod.reshape(-1))                 # [ell_rest·ell_skip, 2]


def _mat_fold(mat_dense, eq):
    """Transposed binary-matrix·vector: out[c] = Σ_{r: M[r,c]=1} eq[r].

    mat_dense: uint64 [k, k] (0/1, indexed [row, col]); eq: [k, 2] -> [k, 2]."""
    sel = mat_dense[:, :, None] * eq[:, None, :]              # M[r,c]·eq[r]  (0/1 select)
    return field.from_ghash(jnp.sum(field.to_ghash(sel), axis=0))  # XOR over rows -> [c, 2]


def fold_alpha_batched(alpha, a_dense, b_dense, eq_inner):
    """comb[c] = α·(A₀ᵀ·eq_inner)[c] ⊕ (B₀ᵀ·eq_inner)[c] (flock
    `sparse_row_fold_alpha_batched`)."""
    ae = field.to_ghash(_mat_fold(a_dense, eq_inner))
    be = field.to_ghash(_mat_fold(b_dense, eq_inner))
    alpha_g = field.to_ghash(jnp.asarray(alpha))
    return field.from_ghash(alpha_g * ae + be)


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

    def fold_alpha_batched(self, alpha, eq_inner):
        eq = jnp.asarray(np.asarray(eq_inner, np.uint64).reshape(-1, 2))
        zero = jnp.zeros((self.k, 2), U64)
        out_a = _seg_xor_fold(eq, *self._a_seg, self.k) if self._a_seg else zero
        out_b = _seg_xor_fold(eq, *self._b_seg, self.k) if self._b_seg else zero
        alpha_g = field.to_ghash(jnp.asarray(alpha))
        return field.from_ghash(alpha_g * field.to_ghash(out_a) + field.to_ghash(out_b))


def partial_fold_packed_z(z_packed_bytes: bytes, m: int, k_log: int, eq_outer):
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
    sel = bits[:, :, None] * eq_outer[:, None, :]                                   # 0/1 select, [n_outer,k,2]
    return field.from_ghash(jnp.sum(field.to_ghash(sel), axis=0))                 # [k, 2]


def _round_eval(c, z):
    """Product-sumcheck round message (q(1), q(∞)) over the TOP-bit split (flock
    `sumcheck_round_eval`): half = len/2; (Σ chi·zhi, Σ (chi+clo)(zhi+zlo))."""
    cg, zg = field.to_ghash(c), field.to_ghash(z)
    half = cg.shape[0] // 2
    clo, chi = cg[:half], cg[half:]
    zlo, zhi = zg[:half], zg[half:]
    e1 = jnp.sum(chi * zhi)
    einf = jnp.sum((chi + clo) * (zhi + zlo))
    return field.from_ghash(e1), field.from_ghash(einf)


def _bind_top(v, r):
    """Bind the top variable at r (flock `sumcheck_bind_top`):
    v'[i] = v[i] + r·(v[i+half] + v[i]); length halves."""
    vg, rg = field.to_ghash(v), field.to_ghash(r)
    half = vg.shape[0] // 2
    vlo, vhi = vg[:half], vg[half:]
    return field.from_ghash(vlo + rg * (vhi + vlo))


@runtime_checkable
class LincheckCircuit(Protocol):
    """The circuit seam `prove` consumes — structural, like zorch's `ProverRound`
    (`zorch.round`): a family of circuits plugged into one unchanging lincheck
    protocol, varying only the column-marginal fold. `fold_alpha_batched` returns
    comb[c] = α·(A₀ᵀ·eq_inner)[c] ⊕ (B₀ᵀ·eq_inner)[c]; `const_pin` is the +β pin
    column, or None. `CscCircuit` (device seg-scan) and the `KeccakLincheckCircuit`
    / `Keccak3LincheckCircuit` host walkers match it structurally — no inheritance,
    no shared base. `@runtime_checkable` so `lincheck_circuit_protocol_test` can
    assert conformance at runtime; that checks member presence, not the fold's math
    (the byte-match oracle gates pin that)."""

    const_pin: int | None

    def fold_alpha_batched(self, alpha: Any, eq_inner: Any) -> Any:
        ...


@dataclass(frozen=True)
class _LincheckCarry:
    """State threaded between lincheck's stage Rounds — inputs plus only what a
    later stage reads from an earlier one. Static config (m, k_log, k_skip,
    capture) lives on the Round instances (cf. zerocheck._ZerocheckCarry). None
    fields are per-stage outputs set via replace; not pytree-registered (no @jit
    boundary)."""

    z_packed_bytes: Any
    a_dense: Any
    b_dense: Any
    x_ab: Any
    circuit: Any
    comb: Any = None                 # ← _CombRound
    rounds: Any = None               # ← _SumcheckRound
    r_rounds: Any = None             # ← _SumcheckRound (read by _ClaimRound)
    z_partial: Any = None            # ← _SumcheckRound
    z_vec_pre: Any = None            # ← _SumcheckRound (capture)
    claim: Any = None                # ← _ClaimRound


class _CombRound(Round):
    """Sample α, build the quirky eq table, and fold the constraint matrices into
    comb = α·(A₀ᵀ·eq_inner) ⊕ (B₀ᵀ·eq_inner) — dense or via a `CscCircuit`, with the
    optional const_pin +β. No proof message — writes comb onto the carry."""

    def __init__(self, k_skip: int):
        self._k_skip = k_skip

    def __call__(self, carry, transcript):
        k_skip = self._k_skip
        x_ab, circuit = carry.x_ab, carry.circuit
        transcript.observe_label(LABEL)
        alpha = jnp.asarray(transcript.sample_f128())
        eq_inner = build_quirky_eq_table(_to_int(x_ab["z_skip"]), x_ab["x_inner_rest"], k_skip)
        if circuit is not None:
            comb = jnp.asarray(circuit.fold_alpha_batched(alpha, eq_inner))
            if circuit.const_pin is not None:
                beta = jnp.asarray(transcript.sample_f128())   # sampled AFTER alpha (flock order)
                col = circuit.const_pin
                comb = comb.at[col].set(
                    field.from_ghash(field.to_ghash(comb[col]) + field.to_ghash(beta)))
        else:
            comb = fold_alpha_batched(alpha, jnp.asarray(carry.a_dense),
                                      jnp.asarray(carry.b_dense), eq_inner)
        return replace(carry, comb=comb), transcript, None


class _SumcheckRound(Round):
    """Partial-fold z at x_outer, then the (k_log − k_skip)-round product sumcheck
    binding the TOP bit. Message = (rounds, z_partial)."""

    def __init__(self, m: int, k_log: int, k_skip: int, capture: bool):
        self._m, self._k_log, self._k_skip, self._capture = m, k_log, k_skip, capture

    def __call__(self, carry, transcript):
        m, k_log, k_skip = self._m, self._k_log, self._k_skip
        inner_rest = k_log - k_skip
        comb = carry.comb
        eq_outer = build_eq_fused(jnp.asarray(carry.x_ab["x_outer"]))
        z_vec = partial_fold_packed_z(carry.z_packed_bytes, m, k_log, eq_outer)
        z_vec_pre = np.asarray(z_vec) if self._capture else None  # pre-sumcheck (PCS open reuse)

        # Unfused on purpose: each round is _round_eval then _bind_top, mirroring
        # flock's steps. Do NOT hand-fuse into Rust's sumcheck_bind_both_and_eval_next
        # — operator fusion is the zkx compiler's job.
        rounds, r_rounds = [], []
        if inner_rest > 0:
            e1, einf = _round_eval(comb, z_vec)
            for t in range(inner_rest):
                transcript.observe_f128(e1)
                transcript.observe_f128(einf)
                r = jnp.asarray(transcript.sample_f128())
                rounds.append((np.asarray(e1), np.asarray(einf)))
                r_rounds.append(r)
                comb = _bind_top(comb, r)
                z_vec = _bind_top(z_vec, r)
                if t + 1 < inner_rest:
                    e1, einf = _round_eval(comb, z_vec)
        z_partial = np.asarray(z_vec)
        carry = replace(carry, rounds=rounds, r_rounds=r_rounds, z_partial=z_partial,
                        z_vec_pre=z_vec_pre)
        return carry, transcript, (rounds, z_partial)


class _ClaimRound(Round):
    """Claim derivation (flock prove_padded_inner steps 6-9): observe z_partial,
    sample a fresh z_skip, then w = ⟨φ8-weights(r_inner_skip), z_partial⟩ and the
    LSB-first r_inner_rest. Only in the capture chain. Message = the claim."""

    def __init__(self, k_skip: int):
        self._k_skip = k_skip

    def __call__(self, carry, transcript):
        k_skip = self._k_skip
        z_partial = carry.z_partial
        transcript.observe_f128_slice(z_partial)              # 6. observe z_partial
        r_inner_skip = transcript.sample_f128()               # 7. fresh z_skip AFTER
        lam = _lagrange_weights(k_skip, _to_int(r_inner_skip), 0)  # 8. φ8 S-domain weights
        lam_arr = jnp.asarray(np.stack([_to_lohi(x) for x in lam]))
        w = field.from_ghash_host(jnp.sum(                         # inner_product
            field.to_ghash(lam_arr) * field.to_ghash(jnp.asarray(z_partial)), axis=0))
        r_inner_rest = [np.asarray(r) for r in reversed(carry.r_rounds)]  # 9. LSB-first
        claim = {"r_inner_skip": np.asarray(r_inner_skip),
                 "r_inner_rest": np.stack(r_inner_rest) if r_inner_rest else np.zeros((0, 2), np.uint64),
                 "w": w}
        return replace(carry, claim=claim), transcript, claim


def lincheck_chain(m: int, k_log: int, k_skip: int, capture: bool) -> ProveChain:
    """The lincheck sub-chain: comb → product sumcheck (→ claim, capture only).
    One definition for the stage wiring (cf. zerocheck.zerocheck_chain). The
    claim derivation is FS-bearing, so it joins the chain only when captured —
    that is the exact transcript difference between the two return shapes."""
    rounds = [_CombRound(k_skip), _SumcheckRound(m, k_log, k_skip, capture)]
    if capture:
        rounds.append(_ClaimRound(k_skip))
    return ProveChain(rounds)


def prove(z_packed_bytes, a_dense, b_dense, x_ab, m, k_log, k_skip,
          domain=b"flock-test-v0", ch=None, capture=False,
          circuit: LincheckCircuit | None = None):
    """Run lincheck. x_ab = dict(z_skip:[2], x_inner_rest:[*,2], x_outer:[*,2]).
    Byte-identical to flock `lincheck::prove`/`prove_padded_capture_z_vec`.

    A `lincheck_chain` of stage `Round`s (comb → sumcheck → claim) threading one
    `Challenger`. `circuit`: a `CscCircuit` for real hash R1CS (sparse A₀/B₀ at
    large k, with an optional const_pin +β column); when None, the dense
    `a_dense`/`b_dense` path is used (small test R1CS). Default returns (rounds,
    z_partial). With `capture=True` (the e2e fused prover) also returns the
    post-sumcheck claim and the pre-sumcheck z_vec. Pass a shared `ch` to thread
    Fiat-Shamir; else a fresh Challenger(domain)."""
    if ch is None:
        ch = Challenger(domain)
    carry, _ch, _msgs = lincheck_chain(m, k_log, k_skip, capture)(
        _LincheckCarry(z_packed_bytes, a_dense, b_dense, x_ab, circuit), ch)
    if not capture:
        return carry.rounds, carry.z_partial
    return carry.rounds, carry.z_partial, carry.claim, carry.z_vec_pre
