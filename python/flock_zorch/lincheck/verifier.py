# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Lincheck verifier (dense A₀/B₀ path).

`ok` is the AND of the rounds' flags: the sumcheck round ties `z_partial` back to
`α·v_a + v_b` through `Σ comb·z_partial`, so an inconsistent `z_partial` fails it.
Native `binary_field_ghash`; the per-round loop is host Python (inner_rest ≤ ~8).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import frx.numpy as fnp

from flock_zorch import ghash
from flock_zorch.lincheck.prover import (
    LABEL,
    LincheckClaim,
    build_quirky_eq_table,
    fold_alpha_batched,
)
from flock_zorch.zerocheck import _lagrange_weights
from zorch.round import Round, VerifyChain
from zorch.sumcheck.domain import fold

_EMPTY_G = ghash.to_ghash(fnp.zeros((0, 2), fnp.uint64))


def _g(x):
    """Any F128 (native ghash or uint64 [..., 2] lanes) → native ghash. Proof
    fields arrive in lane form; challenges are native."""
    return ghash.to_ghash(fnp.asarray(ghash.to_lanes(x)))


@dataclass(frozen=True)
class _VerifyCarry:
    a_dense: Any
    b_dense: Any
    x_ab: Any
    v_a: Any
    v_b: Any
    k_skip: int
    comb: Any = None
    running: Any = None
    r_rounds: Any = None
    z_partial: Any = None
    claim: Any = None


class _CombVerifyRound(Round):
    """Sample α, rebuild comb = α·(A₀ᵀ·eq) ⊕ (B₀ᵀ·eq) from the public matrices, and
    seed the running claim α·v_a + v_b. No message."""

    def __init__(self, k_skip: int):
        self._k_skip = k_skip

    def __call__(self, carry, msg, transcript):
        transcript.observe_label(LABEL)
        alpha = transcript.sample_f128()
        eq_inner = build_quirky_eq_table(carry.x_ab.z_skip, carry.x_ab.x_inner_rest, self._k_skip)
        comb = fold_alpha_batched(
            alpha, fnp.asarray(carry.a_dense), fnp.asarray(carry.b_dense), eq_inner)
        running = alpha * carry.v_a + carry.v_b
        return replace(carry, comb=_g(comb), running=running), transcript, True


class _SumcheckVerifyRound(Round):
    """Replay the ∞-product sumcheck, folding comb at each challenge; `ok` iff the
    reduced running claim equals ⟨comb_partial, z_partial⟩. Message = (rounds, z_partial)."""

    def __call__(self, carry, msg, transcript):
        rounds, z_partial = msg
        z_partial = _g(z_partial)
        comb, running, r_rounds = carry.comb, carry.running, []
        for msg_1, msg_inf in rounds:
            e1, einf = _g(msg_1), _g(msg_inf)
            transcript.observe_f128(e1)
            transcript.observe_f128(einf)
            r = transcript.sample_f128()
            # q(X) = einf·X² + c1·X + e0 through (q(0),q(1),q(∞)); q(0)=claim+q(1) in char 2.
            e0 = running + e1
            c1 = e0 + e1 + einf
            running = einf * r * r + c1 * r + e0
            comb = fold(comb, r)
            r_rounds.append(r)
        ok = running == fnp.sum(comb * z_partial)
        return replace(carry, r_rounds=r_rounds, z_partial=z_partial), transcript, ok


class _ClaimVerifyRound(Round):
    """Observe z_partial, sample the fresh inner z_skip, derive w = ⟨φ8(z_skip), z_partial⟩
    and the LSB-first inner-rest challenges. No message."""

    def __init__(self, k_skip: int):
        self._k_skip = k_skip

    def __call__(self, carry, msg, transcript):
        transcript.observe_f128(carry.z_partial)
        r_inner_skip = transcript.sample_f128()
        w = fnp.sum(_lagrange_weights(self._k_skip, r_inner_skip, 0) * carry.z_partial)
        rev = list(reversed(carry.r_rounds))
        r_inner_rest = fnp.stack(rev) if rev else _EMPTY_G
        claim = LincheckClaim(r_inner_skip=r_inner_skip, r_inner_rest=r_inner_rest, w=w)
        return replace(carry, claim=claim), transcript, True


def lincheck_verify_chain(k_skip: int) -> VerifyChain:
    """comb → product sumcheck → claim, the verify side of `lincheck_chain`."""
    return VerifyChain([_CombVerifyRound(k_skip), _SumcheckVerifyRound(), _ClaimVerifyRound(k_skip)])


def verify(m, k_log, k_skip, a_dense, b_dense, x_ab, v_a, v_b, proof, transcript):
    """Verify a `LincheckProof` given the zerocheck's â/b̂ evals `v_a`, `v_b` and the
    shared point `x_ab`. Returns `(LincheckClaim, transcript, ok)`; raises `ValueError`
    on a malformed proof shape."""
    inner_rest = k_log - k_skip
    if len(proof.rounds) != inner_rest:
        raise ValueError(f"expected {inner_rest} sumcheck rounds, got {len(proof.rounds)}")
    if proof.z_partial.shape[0] != (1 << k_skip):
        raise ValueError(f"z_partial must have length 2^k_skip = {1 << k_skip}")

    carry = _VerifyCarry(a_dense, b_dense, x_ab, _g(v_a), _g(v_b), k_skip)
    msgs = [None, (proof.rounds, proof.z_partial), None]
    carry, transcript, ok = lincheck_verify_chain(k_skip)(carry, msgs, transcript)
    return carry.claim, transcript, ok
