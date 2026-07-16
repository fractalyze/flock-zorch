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
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
import frx
import frx.numpy as jnp

from flock_zorch import ghash
from flock_zorch.sumcheck import build_eq_fused, ONE
from flock_zorch.zerocheck import _lagrange_weights, ZerocheckProof
from flock_zorch.challenger import Challenger
from flock_zorch.lincheck._csc_fold import _flatten_nz, _csc_segments, _seg_xor_fold
from flock_zorch.sumcheck.inf_product import prove_inf_product
from zorch.round import ProveChain, Round

U64 = jnp.uint64
LABEL = b"flock-lincheck-v0"
_ZERO_G = frx.lax.bitcast_convert_type(jnp.zeros(2, U64), jnp.binary_field_ghash)


def build_quirky_eq_table(z_skip, x_inner_rest, k_skip: int):
    """eq_inner[i_skip + i_rest·2^k_skip] = λ_skip[i_skip]·eq_rest[i_rest]
    (flock `build_quirky_eq_table`; i_skip in the LOW bits). z_skip: ghash scalar
    (the zerocheck fold point). Returns the eq table as native ghash [ell_rest·ell_skip]."""
    lam = _lagrange_weights(k_skip, z_skip, 0)
    eq_rest = build_eq_fused(x_inner_rest)             # ghash coords -> [ell_rest]
    prod = eq_rest[:, None] * lam[None, :]                    # [ell_rest, ell_skip]
    return prod.reshape(-1)


def _mat_fold(mat_dense, eq):
    """Transposed binary-matrix·vector: out[c] = Σ_{r: M[r,c]=1} eq[r].

    mat_dense: uint64 [k, k] (0/1, indexed [row, col]); eq: ghash [k] -> ghash [k].
    The 0/1 marginal is a dtype-native select (mask · ghash isn't a field mul)."""
    return jnp.sum(jnp.where(mat_dense.astype(bool), eq[:, None], _ZERO_G), axis=0)


def fold_alpha_batched(alpha, a_dense, b_dense, eq_inner):
    """comb[c] = α·(A₀ᵀ·eq_inner)[c] ⊕ (B₀ᵀ·eq_inner)[c] (flock
    `sparse_row_fold_alpha_batched`)."""
    ae = _mat_fold(a_dense, eq_inner)
    be = _mat_fold(b_dense, eq_inner)
    return alpha * ae + be


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
        eq = jnp.asarray(eq_inner).reshape(-1)                # ghash [k]
        zero = frx.lax.bitcast_convert_type(jnp.zeros((self.k, 2), U64), jnp.binary_field_ghash)
        out_a = _seg_xor_fold(eq, *self._a_seg, self.k) if self._a_seg else zero
        out_b = _seg_xor_fold(eq, *self._b_seg, self.k) if self._b_seg else zero
        return alpha * out_a + out_b


def partial_fold_packed_z(z_packed_bytes: bytes, m: int, k_log: int, eq_outer):
    """z_vec[i_inner] = Σ_{i_outer} z(i_inner, i_outer)·eq_outer[i_outer]
    (flock `partial_fold_packed_z`, useful_bits = 2^k_log).

    z_packed layout: byte `z_packed[byte_idx·k + i_inner]` holds outer bits
    `z[i_inner, 8·byte_idx + r]` at bit r."""
    k = 1 << k_log
    n_outer = 1 << (m - k_log)
    n_bytes = n_outer // 8
    zp = jnp.asarray(np.frombuffer(z_packed_bytes, np.uint8).reshape(n_bytes, k))
    return _partial_fold(zp, eq_outer, n_outer)         # device + jit (keeps the intermediate off HBM)


@functools.partial(frx.jit, static_argnums=(2,))
def _partial_fold(zp, eq_outer, n_outer):
    """z_vec[i_inner] = Σ_{i_outer} bit·eq_outer[i_outer], device+jit so the large
    [n_outer,k,2] intermediate stays fused on device and never lands in HBM."""
    bits = ((zp[:, None, :] >> jnp.arange(8, dtype=jnp.uint8)[None, :, None]) & 1)  # [nb,8,k]
    bits = bits.reshape(n_outer, zp.shape[1]).astype(bool)                          # i_outer=byte·8+r
    return jnp.sum(jnp.where(bits, eq_outer[:, None], _ZERO_G), axis=0)  # dtype-native 0/1 select


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
class AbClaimPoint:
    """The â/b̂ evaluation point lincheck reduces — the zerocheck challenge split
    (flock's QuirkyPoint): `z_skip` the URM fold-point, `x_inner_rest` the inner
    multilinear challenges, `x_outer` the outer ones."""

    z_skip: Any
    x_inner_rest: Any
    x_outer: Any

    @classmethod
    def from_zerocheck(cls, zc: ZerocheckProof, inner_rest: int) -> "AbClaimPoint":
        """The â/b̂ point derived from a zerocheck proof: z_skip is the URM
        fold-point, and the multilinear challenges split into inner/outer at
        `inner_rest`."""
        return cls(z_skip=zc.z,
                   x_inner_rest=zc.mlv_challenges[:inner_rest],
                   x_outer=zc.mlv_challenges[inner_rest:])


@dataclass(frozen=True)
class LincheckClaim:
    """The post-sumcheck claim (flock prove_padded_inner steps 6-9): the fresh
    inner z_skip, the LSB-first inner-rest challenges, and the reduced value w."""

    r_inner_skip: Any
    r_inner_rest: Any
    w: Any


class LincheckProof(NamedTuple):
    """flock's lincheck proof: the product-sumcheck `rounds` and the `z_partial`
    message. `claim` (a `LincheckClaim`) and `z_vec_pre` are populated only on the
    captured (e2e) path — the post-sumcheck claim and the pre-sumcheck z_vec the
    PCS open reuses — and are None otherwise. A NamedTuple (not a dataclass) so the
    historical `rounds, z_partial, claim, z_vec_pre = prove(...)` unpacking keeps
    working alongside attribute access."""

    rounds: Any
    z_partial: Any
    claim: "LincheckClaim | None" = None
    z_vec_pre: Any = None


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
    z_partial: Any = None            # ← _SumcheckRound (lanes, for the wire)
    z_partial_g: Any = None          # ← _SumcheckRound (for observe + w, no host lift)
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
        alpha = transcript.sample_f128()
        eq_inner = build_quirky_eq_table(x_ab.z_skip, x_ab.x_inner_rest, k_skip)
        if circuit is not None:
            comb = circuit.fold_alpha_batched(alpha, eq_inner)
            if circuit.const_pin is not None:
                beta = transcript.sample_f128()               # sampled AFTER alpha (flock order)
                col = circuit.const_pin
                comb = comb.at[col].set(comb[col] + beta)
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
        eq_outer = build_eq_fused(carry.x_ab.x_outer)
        z_vec = partial_fold_packed_z(carry.z_packed_bytes, m, k_log, eq_outer)
        z_vec_pre = ghash.from_ghash_host(z_vec) if self._capture else None  # pre-sumcheck (PCS open reuse)

        rounds, r_rounds = [], []
        if inner_rest > 0:
            stacked = jnp.stack([comb, z_vec])
            stacked, transcript._t, msgs = prove_inf_product(
                stacked, transcript._t, inner_rest)
            for e1, einf, r in msgs:
                rounds.append((e1, einf))
                r_rounds.append(r)                            # native ghash fold challenge
            z_partial_g = stacked[1]
        else:
            z_partial_g = z_vec
        z_partial = z_partial_g
        carry = replace(carry, rounds=rounds, r_rounds=r_rounds, z_partial=z_partial,
                        z_partial_g=z_partial_g, z_vec_pre=z_vec_pre)
        return carry, transcript, (rounds, z_partial)


class _ClaimRound(Round):
    """Claim derivation (flock prove_padded_inner steps 6-9): observe z_partial,
    sample a fresh z_skip, then w = ⟨φ8-weights(r_inner_skip), z_partial⟩ and the
    LSB-first r_inner_rest. Only in the capture chain. Message = the claim."""

    def __init__(self, k_skip: int):
        self._k_skip = k_skip

    def __call__(self, carry, transcript):
        k_skip = self._k_skip
        transcript.observe_f128(carry.z_partial_g)      # 6. observe z_partial
        r_inner_skip = transcript.sample_f128()               # 7. fresh z_skip AFTER
        lam = _lagrange_weights(k_skip, r_inner_skip, 0)       # 8. φ8 S-domain weights
        w = jnp.sum(lam * carry.z_partial_g, axis=0)          # inner_product (ghash)
        r_inner_rest = list(reversed(carry.r_rounds))         # 9. LSB-first (ghash scalars)
        claim = LincheckClaim(
            r_inner_skip=r_inner_skip,
            r_inner_rest=(jnp.stack(r_inner_rest) if r_inner_rest
                          else ghash.to_ghash(jnp.zeros((0, 2), jnp.uint64))),
            w=w)
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


def prove(z_packed_bytes, a_dense, b_dense, x_ab: AbClaimPoint, m: int, k_log: int,
          k_skip: int, domain: bytes = b"flock-test-v0", ch: Challenger | None = None,
          capture: bool = False, circuit: LincheckCircuit | None = None) -> LincheckProof:
    """Run lincheck. `x_ab` is an `AbClaimPoint` (z_skip:[2], x_inner_rest:[*,2],
    x_outer:[*,2]). Byte-identical to flock `lincheck::prove`/`prove_padded_capture_z_vec`.

    A `lincheck_chain` of stage `Round`s (comb → sumcheck → claim) threading one
    `Challenger`. `circuit`: a `CscCircuit` for real hash R1CS (sparse A₀/B₀ at
    large k, with an optional const_pin +β column); when None, the dense
    `a_dense`/`b_dense` path is used (small test R1CS). Returns a `LincheckProof`;
    its `claim`/`z_vec_pre` are populated only with `capture=True` (the e2e fused
    prover). Pass a shared `ch` to thread Fiat-Shamir; else a fresh Challenger(domain)."""
    if ch is None:
        ch = Challenger(domain)
    carry, _ch, _msgs = lincheck_chain(m, k_log, k_skip, capture)(
        _LincheckCarry(z_packed_bytes, a_dense, b_dense, x_ab, circuit), ch)
    return LincheckProof(rounds=carry.rounds, z_partial=carry.z_partial,
                         claim=carry.claim, z_vec_pre=carry.z_vec_pre)
