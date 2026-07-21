# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Zerocheck verifier.

Soundness is the AND of the rounds' `ok` flags (`VerifyChain`): a failed check
yields `ok=False`; only a malformed proof shape raises. Arithmetic is native
`binary_field_ghash`; the per-variable loop is host Python (n_mlv ≤ ~20).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import frx.numpy as fnp

from flock_zorch import ghash
from flock_zorch.zerocheck import _urm
from flock_zorch.zerocheck._fold import _batch_inv, _lag_numden, _lag_w
from flock_zorch.zerocheck.prover import K_SKIP, LABEL, N_INNER, _MEDIUM_G, _SMALL_G
from zorch.round import Round, VerifyChain

_ONE_G = ghash.to_ghash(fnp.array([1, 0], fnp.uint64))


def _inv(x):
    """GF(2^128) inverse of a native-ghash scalar (`_batch_inv` wants a 1-D array)."""
    return _batch_inv(fnp.reshape(x, (1,)))[0]


def _g(x):
    """Any F128 — native `binary_field_ghash` or uint64 `[..., 2]` lanes — to native
    ghash. Proof fields arrive in lane form; transcript challenges are native, so
    normalize before mixing them."""
    return ghash.to_ghash(fnp.asarray(ghash.to_lanes(x)))


def _lagrange_at_z(nodes_g, values_g, zg):
    """Σ_i L_i(z)·values[i] over `nodes_g` (native ghash) → native ghash scalar.
    `values_g` aligns with the LAST len(values_g) nodes: the Λ half of the combined
    Λ∪S set, or all of the plain Λ set."""
    num, den = _lag_numden(nodes_g, zg)
    w = _lag_w(num, _batch_inv(den))
    return fnp.sum(w[nodes_g.shape[0] - values_g.shape[0]:] * values_g)


@dataclass(frozen=True)
class ZerocheckClaim:
    """flock `ZerocheckClaim`: the evaluation point (`z` skip-scalar + `mlv_challenges`
    / `r_rest` coordinate lists) and the final â/b̂/ĉ evals bound at it."""

    z: Any
    mlv_challenges: Any
    r_rest: Any
    a_eval: Any
    b_eval: Any
    c_eval: Any


@dataclass(frozen=True)
class _VerifyCarry:
    """Threaded verifier state — the reconstructed claim, filled round by round."""

    r_rest: Any = None
    z: Any = None
    c_running: Any = None
    mlv_challenges: Any = None
    a_eval: Any = None
    b_eval: Any = None
    c_eval: Any = None


class _SetupVerifyRound(Round):
    """Re-derive r = skip challenges ++ the fixed inner-7 constants ++ outer
    challenges. No message; always `ok`."""

    def __init__(self, m: int, k_skip: int):
        self._m, self._k_skip = m, k_skip

    def __call__(self, carry, msg, transcript):
        transcript.observe_label(LABEL)
        transcript.sample_f128(self._k_skip)  # r_skip: advance FS, not a "rest" coord
        r_outer = transcript.sample_f128(self._m - self._k_skip - N_INNER)
        r_rest = fnp.concatenate([_SMALL_G, _MEDIUM_G, r_outer])  # r[k_skip:]
        return replace(carry, r_rest=r_rest), transcript, True


class _UrmVerifyRound(Round):
    """Reconstruct ĉ(z) and the AB running claim from round-1. Message =
    (round1_ab, round1_c, final_c_eval); `ok` iff the sent ĉ equals the
    re-derivation."""

    def __init__(self, m: int, k_skip: int):
        self._m, self._k_skip = m, k_skip

    def __call__(self, carry, msg, transcript):
        ell = 1 << self._k_skip
        ab, c, final_c = _g(msg[0]), _g(msg[1]), _g(msg[2])
        transcript.observe_f128(ab)
        transcript.observe_f128(c)
        z = transcript.sample_f128()

        lam = ghash.to_ghash(fnp.asarray(_urm.PHI_8_TABLE[ell:2 * ell]))  # Λ nodes
        full = ghash.to_ghash(fnp.asarray(_urm.PHI_8_TABLE[:2 * ell]))    # Λ∪S nodes
        p_c_at_z = _lagrange_at_z(lam, c, z)
        # AB running claim: interpolate the combined poly (zero on S) over Λ∪S at z,
        # then P^{AB}(z) = combined(z) + P^C(z).
        c_running = _lagrange_at_z(full, ab + c, z) + p_c_at_z

        ok = p_c_at_z == final_c
        return replace(carry, z=z, c_running=c_running, c_eval=final_c), transcript, ok


class _MultilinearVerifyRound(Round):
    """Sumcheck consistency chain: reconstruct G(0) from the running claim, fold at
    ρ via the char-2 quadratic G(X)=G0·(1+X)+G1·X+G∞·X·(1+X), bind â/b̂. Message =
    (rounds, final_a, final_b); `ok` iff the final running claim equals â·b̂."""

    def __init__(self, m: int, k_skip: int):
        self._m, self._k_skip = m, k_skip

    def __call__(self, carry, msg, transcript):
        rounds, final_a, final_b = msg
        final_a, final_b = _g(final_a), _g(final_b)
        c_running = carry.c_running
        rhos = []
        for i, (msg_1, msg_inf) in enumerate(rounds):
            g1, g_inf = _g(msg_1), _g(msg_inf)
            r_eq = carry.r_rest[i]
            g0 = (c_running + r_eq * g1) * _inv(_ONE_G + r_eq)
            transcript.observe_f128(g1)
            transcript.observe_f128(g_inf)
            rho = transcript.sample_f128()
            rhos.append(rho)
            one_plus_rho = _ONE_G + rho
            c_running = g0 * one_plus_rho + g1 * rho + g_inf * rho * one_plus_rho

        ok = c_running == final_a * final_b
        transcript.observe_f128(final_a)  # bind finals at the prover's position (before α)
        transcript.observe_f128(final_b)
        carry = replace(carry, mlv_challenges=fnp.stack(rhos), a_eval=final_a, b_eval=final_b)
        return carry, transcript, ok


def zerocheck_verify_chain(m: int, k_skip: int) -> VerifyChain:
    """setup → round-1 URM → multilinear sumcheck, the verify side of `zerocheck_chain`."""
    return VerifyChain([_SetupVerifyRound(m, k_skip), _UrmVerifyRound(m, k_skip),
                        _MultilinearVerifyRound(m, k_skip)])


def verify(m: int, proof, transcript):
    """Verify a `ZerocheckProof` against `m`, threading `transcript`. Returns
    `(ZerocheckClaim, transcript, ok)`; `ok` is False for an unsound proof. Raises
    `ValueError` only on a structurally malformed proof (wrong message lengths)."""
    k_skip = K_SKIP
    if m < k_skip + N_INNER:
        raise ValueError(f"log_n {m} < k_skip + N_INNER = {k_skip + N_INNER}")
    ell = 1 << k_skip
    if proof.round1_ab.shape[0] != ell or proof.round1_c.shape[0] != ell:
        raise ValueError(f"round-1 messages must have length ell={ell}")
    if len(proof.multilinear_rounds) != m - k_skip:
        raise ValueError(f"expected {m - k_skip} multilinear rounds")

    msgs = [
        None,
        (proof.round1_ab, proof.round1_c, proof.final_c_eval),
        (proof.multilinear_rounds, proof.final_a_eval, proof.final_b_eval),
    ]
    carry, transcript, ok = zerocheck_verify_chain(m, k_skip)(_VerifyCarry(), msgs, transcript)
    claim = ZerocheckClaim(
        z=carry.z, mlv_challenges=carry.mlv_challenges, r_rest=carry.r_rest,
        a_eval=carry.a_eval, b_eval=carry.b_eval, c_eval=carry.c_eval,
    )
    return claim, transcript, ok
