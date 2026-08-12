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

import frx
import frx.numpy as fnp
import numpy as np
from zorch.round import prove_rounds
from zorch.stage import ProveResult, ProverStage

from flock_zorch import ghash
from flock_zorch.lincheck._csc_fold import _csc_segments, _flatten_nz, _seg_xor_fold
from flock_zorch.sha256_challenger import Sha256Challenger
from flock_zorch.sumcheck import build_eq
from flock_zorch.sumcheck.inf_product import prove_inf_product
from flock_zorch.types import (
    AbClaimPoint,
    BatchOpeningClaim,
    LincheckClaim,
    LincheckProof,
    R1csWitness,
    ZerocheckClaim,
)
from flock_zorch.zerocheck import _lagrange_weights

U64 = fnp.uint64
_GHASH = fnp.binary_field_ghash
LABEL = b"flock-lincheck-v0"


@functools.partial(frx.jit, static_argnums=(2,))
def build_quirky_eq_table(z_skip, x_inner_rest, k_skip: int):
    """eq_inner[i_skip + i_rest·2^k_skip] = λ_skip[i_skip]·eq_rest[i_rest]
    (flock `build_quirky_eq_table`; i_skip in the LOW bits). z_skip: ghash scalar
    (the zerocheck fold point). Returns the eq table as native ghash
    [ell_rest·ell_skip].

    One jitted kernel (k_skip static), so the Lagrange weights and `build_eq` (called
    directly, not `build_eq_fused`) fuse — the eager caller gets one fused build."""
    lam = _lagrange_weights(k_skip, z_skip, 0)
    eq_rest = build_eq(x_inner_rest)  # ghash coords -> [ell_rest]
    prod = eq_rest[:, None] * lam[None, :]  # [ell_rest, ell_skip]
    return prod.reshape(-1)


def _mat_fold(mat_dense, eq):
    """Transposed binary-matrix·vector: out[c] = Σ_{r: M[r,c]=1} eq[r].

    mat_dense: uint64 [k, k] (0/1, indexed [row, col]); eq: ghash [k] -> ghash [k].
    The 0/1 marginal is a dtype-native select (mask · ghash isn't a field mul)."""
    return fnp.sum(
        fnp.where(mat_dense.astype(bool), eq[:, None], fnp.zeros((), _GHASH)), axis=0
    )


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
        eq = fnp.asarray(eq_inner).reshape(-1)  # ghash [k]
        zero = fnp.zeros(self.k, _GHASH)
        out_a = _seg_xor_fold(eq, *self._a_seg, self.k) if self._a_seg else zero
        out_b = _seg_xor_fold(eq, *self._b_seg, self.k) if self._b_seg else zero
        return alpha * out_a + out_b


def stripe_to_device(z_packed_bytes: bytes | bytearray, m: int, k_log: int):
    """Upload the packed lincheck stripe as the device-resident [n_bytes, k]
    uint8 array `partial_fold_packed_z` consumes. Call once at witness ingest,
    alongside the other witness buffers — the stripe is per-witness, and left
    as host bytes it re-crosses PCIe on every prove (512 MiB at m=32)."""
    k = 1 << k_log
    n_bytes = (1 << (m - k_log)) // 8
    return fnp.asarray(np.frombuffer(z_packed_bytes, np.uint8).reshape(n_bytes, k))


def partial_fold_packed_z(z_stripe, m: int, k_log: int, x_outer):
    """z_vec[i_inner] = Σ_{i_outer} z(i_inner, i_outer)·eq_outer[i_outer]
    (flock `partial_fold_packed_z`, useful_bits = 2^k_log). `x_outer` is the outer
    challenge; the eq table is built inside the jitted core.

    `z_stripe` is either the host packed bytes (uploaded here, paying PCIe per
    call) or the device array `stripe_to_device` returns.

    z_packed layout: byte `z_packed[byte_idx·k + i_inner]` holds outer bits
    `z[i_inner, 8·byte_idx + r]` at bit r."""
    n_outer = 1 << (m - k_log)
    if isinstance(z_stripe, (bytes, bytearray)):
        z_stripe = stripe_to_device(z_stripe, m, k_log)
    elif z_stripe.shape != (n_outer // 8, 1 << k_log):
        raise ValueError(
            f"device stripe shape {z_stripe.shape} != {(n_outer // 8, 1 << k_log)}"
        )
    return _partial_fold(
        z_stripe, x_outer, n_outer
    )  # device + jit (keeps the intermediate off HBM)


@functools.partial(frx.jit, static_argnums=(2,))
def _partial_fold(zp, x_outer, n_outer):
    """z_vec[i_inner] = Σ_{i_outer} bit·eq_outer[i_outer], device+jit so the large
    [n_outer,k,2] intermediate stays fused on device and never lands in HBM.

    The barrier makes `build_eq` materialize its [n_outer] table (4 MiB at
    m=32) before the reduce. Without it XLA fuses the tail of the eq hypercube
    expansion into the reduce fusion and recomputes eq per reduce-input
    element — a clmul chain that deepens with every doubling past the
    materialized [2^14] prefix, which is what made the fold superlinear
    (124 ms instead of 3.4 ms at m=32, blake3 shape)."""
    eq_outer = frx.lax.optimization_barrier(build_eq(x_outer))
    bits = (
        zp[:, None, :] >> fnp.arange(8, dtype=fnp.uint8)[None, :, None]
    ) & 1  # [nb,8,k]
    bits = bits.reshape(n_outer, zp.shape[1]).astype(bool)  # i_outer=byte·8+r
    return fnp.sum(
        fnp.where(bits, eq_outer[:, None], fnp.zeros((), _GHASH)), axis=0
    )  # dtype-native 0/1 select


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

    def fold_alpha_batched(self, alpha: Any, eq_inner: Any) -> Any: ...


@dataclass(frozen=True)
class _LincheckCarry:
    """State threaded between the lincheck's steps — inputs plus only what a later
    step reads from an earlier one. Static config (m, k_log, k_skip) lives on the
    step instances (cf. zerocheck._ZerocheckCarry); the `None` fields are per-step
    outputs set via `replace`. Not pytree-registered (no @jit boundary)."""

    z_stripe: Any
    a_dense: Any
    b_dense: Any
    x_ab: Any
    circuit: Any
    comb: Any = None  # ← _CombRound
    rounds: Any = None  # ← _SumcheckRound
    r_rounds: Any = None  # ← _SumcheckRound (read by _ClaimRound)
    z_partial: Any = None  # ← _SumcheckRound (native ghash; observe + w + wire)
    claim: Any = None  # ← _ClaimRound


class _CombRound:
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
                beta = transcript.sample_f128()  # sampled AFTER alpha (flock order)
                col = circuit.const_pin
                comb = comb.at[col].set(comb[col] + beta)
        else:
            comb = fold_alpha_batched(
                alpha, fnp.asarray(carry.a_dense), fnp.asarray(carry.b_dense), eq_inner
            )
        return replace(carry, comb=comb), transcript, None


class _SumcheckRound:
    """Partial-fold z at x_outer, then the (k_log − k_skip)-round product sumcheck
    binding the TOP bit. Message = (rounds, z_partial)."""

    def __init__(self, m: int, k_log: int, k_skip: int):
        self._m, self._k_log, self._k_skip = m, k_log, k_skip

    def __call__(self, carry, transcript):
        m, k_log, k_skip = self._m, self._k_log, self._k_skip
        inner_rest = k_log - k_skip
        comb = carry.comb
        z_vec = partial_fold_packed_z(carry.z_stripe, m, k_log, carry.x_ab.x_outer)

        rounds, r_rounds = [], []
        if inner_rest > 0:
            stacked = fnp.stack([comb, z_vec])
            stacked, transcript._t, msgs = prove_inf_product(
                stacked, transcript._t, inner_rest
            )
            for e1, einf, r in msgs:
                rounds.append((e1, einf))
                r_rounds.append(r)  # native ghash fold challenge
            z_partial = stacked[1]
        else:
            z_partial = z_vec
        carry = replace(carry, rounds=rounds, r_rounds=r_rounds, z_partial=z_partial)
        return carry, transcript, (rounds, z_partial)


class _ClaimRound:
    """Claim derivation (flock prove_padded_inner steps 6-9): observe z_partial,
    sample a fresh z_skip, then w = ⟨φ8-weights(r_inner_skip), z_partial⟩ and the
    LSB-first r_inner_rest. FS-bearing (one observe + one draw). Message = the
    claim."""

    def __init__(self, k_skip: int):
        self._k_skip = k_skip

    def __call__(self, carry, transcript):
        k_skip = self._k_skip
        transcript.observe_f128(carry.z_partial)  # 6. observe z_partial
        r_inner_skip = transcript.sample_f128()  # 7. fresh z_skip AFTER
        lam = _lagrange_weights(k_skip, r_inner_skip, 0)  # 8. φ8 S-domain weights
        w = fnp.sum(lam * carry.z_partial, axis=0)  # inner_product (ghash)
        r_inner_rest = list(reversed(carry.r_rounds))  # 9. LSB-first (ghash scalars)
        claim = LincheckClaim(
            r_inner_skip=r_inner_skip,
            r_inner_rest=(
                fnp.stack(r_inner_rest)
                if r_inner_rest
                else ghash.to_ghash(fnp.zeros((0, 2), fnp.uint64))
            ),
            w=w,
        )
        return replace(carry, claim=claim), transcript, claim


def lincheck_steps(m: int, k_log: int, k_skip: int) -> list:
    """The lincheck sub-sequence: comb → product sumcheck → claim. One definition
    for the wiring (cf. zerocheck.zerocheck_steps). Drive with
    ``zorch.round.prove_rounds``."""
    return [_CombRound(k_skip), _SumcheckRound(m, k_log, k_skip), _ClaimRound(k_skip)]


def prove(
    z_stripe,
    a_dense,
    b_dense,
    x_ab: AbClaimPoint,
    m: int,
    k_log: int,
    k_skip: int,
    domain: bytes = b"flock-test-v0",
    ch: Sha256Challenger | None = None,
    circuit: LincheckCircuit | None = None,
) -> LincheckProof:
    """Run lincheck. `x_ab` is an `AbClaimPoint` (z_skip:[2], x_inner_rest:[*,2],
    x_outer:[*,2]). Byte-identical to flock `prove_padded_capture_z_vec` — the
    claim-bearing entry point, the only one anything downstream consumes.
    `z_stripe`: host packed bytes or the device array `stripe_to_device`
    returns (see `partial_fold_packed_z`).

    A `lincheck_steps` sequence (comb → sumcheck → claim) threading one
    `Sha256Challenger`. `circuit`: a `CscCircuit` for real hash R1CS (sparse A₀/B₀ at
    large k, with an optional const_pin +β column); when None, the dense
    `a_dense`/`b_dense` path is used (small test R1CS). Returns a `LincheckProof`.
    Pass a shared `ch` to thread Fiat-Shamir; else a fresh Sha256Challenger(domain)."""
    if ch is None:
        ch = Sha256Challenger(domain)
    carry, _ch, _msgs = prove_rounds(
        lincheck_steps(m, k_log, k_skip),
        _LincheckCarry(z_stripe, a_dense, b_dense, x_ab, circuit),
        ch,
    )
    return LincheckProof(
        rounds=carry.rounds, z_partial=carry.z_partial, claim=carry.claim
    )


class LincheckProver(ProverStage[ZerocheckClaim, R1csWitness, BatchOpeningClaim, Any]):
    """Reduce a = A·ẑ and b = B·ẑ to a single ab evaluation claim on ẑ, and pair
    it with the zerocheck's c claim for the batched open."""

    def __init__(self, m, k_log, k_skip, circuit=None):
        self._m, self._k_log, self._k_skip, self._circuit = m, k_log, k_skip, circuit

    def prove(
        self, claim: ZerocheckClaim, witness: R1csWitness, transcript
    ) -> ProveResult[BatchOpeningClaim, Any]:
        inner_rest = self._k_log - self._k_skip
        x_outer = claim.mlv_challenges[inner_rest:]
        lp = prove(
            witness.z_lincheck,
            witness.a0,
            witness.b0,
            AbClaimPoint(
                z_skip=claim.z,
                x_inner_rest=claim.mlv_challenges[:inner_rest],
                x_outer=x_outer,
            ),
            self._m,
            self._k_log,
            self._k_skip,
            ch=transcript,
            circuit=self._circuit,
        )
        if lp.claim is None:
            # `LincheckProof.claim` is optional only to keep the historical
            # `rounds, z_partial, claim` unpacking working; a prove that reached
            # here without one cannot state what it reduced to.
            raise ValueError("lincheck produced no claim to open against")
        # c_full is split-then-rejoined (not just r_rest) to mirror Rust's
        # QuirkyPoint / quirky_x_outer_full.
        return ProveResult(
            BatchOpeningClaim(
                ab_point=fnp.concatenate([lp.claim.r_inner_rest, x_outer], axis=0),
                c_point=fnp.concatenate(
                    [claim.r_rest[:inner_rest], claim.r_rest[inner_rest:]], axis=0
                ),
                ab_value=lp.claim.w,
                c_value=claim.c_eval,
            ),
            (lp.rounds, lp.z_partial),
            transcript,
        )
