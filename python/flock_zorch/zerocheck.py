"""flock's zerocheck `prove_packed` — the first FULL PIOP sub-protocol with a
serializable proof — authored as a host round loop, byte-identical to flock-core's
`zerocheck::prove_packed_padded_inner`.

Proves `a(y)·b(y) ⊕ c(y) = 0 ∀ y ∈ {0,1}^m`. Structure: one univariate-skip
round-1 (URM, `gf8.round1_naive`) over K_SKIP=6 skip variables, then a multilinear
sumcheck over the remaining `m − K_SKIP` variables (the iter-10 `sumcheck`
primitives). Fiat-Shamir is the host SHA-256 `Challenger`; the bulk field arith
(`round_pair`/`fold_pair`) runs on the native `binary_field_ghash` multiply (→ clmad on GPU).

The protocol fixes the inner 7 of the `r` challenge coordinates to constants
(`small`/`medium`), and the C track is pinned at round 1 (extract_c), so only AB
participate in the multilinear rounds — `final_c_eval` is an interpolation of
`round1_c` at the URM fold-point `z`. Requires `jax_enable_x64` and `zorch` on
PYTHONPATH.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from flock_zorch import sumcheck
from flock_zorch.field import gf8
from flock_zorch.field import _to_int, _to_lohi
from flock_zorch.field import _hostfield as hf
from flock_zorch.challenger import Challenger
from flock_zorch._zerocheck_fold import (
    _lagrange_weights, _interpolate_at_z_on_lambda, _fold_at_z_rows, _phi_int, _ONE,
)
from zorch.round import ProveChain, Round

K_SKIP = 6
N_INNER = 7  # 3 small + 4 medium fixed-constant inner dims
LABEL = b"flock-zerocheck-v0"


@dataclass(frozen=True)
class _ZerocheckCarry:
    """State threaded between zerocheck's stage Rounds — only what a later stage
    reads from an earlier one. Static config (m, k_skip) lives on the Round
    instances (cf. prover._ProveCarry). None fields are per-stage outputs set via
    replace. Not pytree-registered: the chain never crosses a @jit boundary (the
    per-round field ops jit internally via _jit_round_fold)."""

    a_bits: Any
    b_bits: Any
    c_bits: Any
    r: Any = None                    # ← _SetupRound
    a_rows: Any = None               # ← _UrmRound (reused by _MultilinearRound)
    b_rows: Any = None               # ← _UrmRound
    round1_ab: Any = None            # ← _UrmRound
    round1_c: Any = None             # ← _UrmRound
    z: Any = None                    # ← _UrmRound
    final_c_eval: Any = None         # ← _UrmRound
    multilinear_rounds: Any = None   # ← _MultilinearRound
    final_a_eval: Any = None         # ← _MultilinearRound
    final_b_eval: Any = None         # ← _MultilinearRound
    mlv_challenges: Any = None       # ← _MultilinearRound


def small_challenges() -> list[int]:
    """[φ₈(0xF7), φ₈(0x53), φ₈(0xB5)] (flock `small_challenges_ghash`)."""
    return [_phi_int(0xF7), _phi_int(0x53), _phi_int(0xB5)]


def medium_challenges() -> list[int]:
    """[γ^E·(1+γ^E)⁻¹ for E∈{1,2,4,8}], γ^E = single bit at lo position E
    (flock `medium_challenges_ghash`)."""
    out = []
    for e in (1, 2, 4, 8):
        ge = 1 << e
        out.append(hf.mul(ge, hf.inv(1 ^ ge)))
    return out


# Module-level jit memo for the per-round field ops. Defining these ONCE (not per
# prove_packed call) lets jax reuse compiled kernels across proofs and across
# rounds of the same shape — otherwise every call makes fresh lambdas and
# recompiles all n_mlv round kernels from scratch.
_JIT_ROUND_FOLD = None


def _jit_round_fold():
    global _JIT_ROUND_FOLD
    if _JIT_ROUND_FOLD is None:
        _JIT_ROUND_FOLD = (jax.jit(lambda a, b, rr: sumcheck.round_pair(a, b, rr)),
                           jax.jit(lambda a, b, rr: sumcheck.fold_pair(a, b, rr)))
    return _JIT_ROUND_FOLD


class _SetupRound(Round):
    """Sample the challenge vector r and fix the inner-7 constants (small ++
    medium). No proof message — writes r onto the carry."""

    def __init__(self, m: int, k_skip: int):
        self._m, self._k_skip = m, k_skip

    def __call__(self, carry, transcript):
        m, k_skip = self._m, self._k_skip
        transcript.observe_label(LABEL)
        r_skip = transcript.sample_f128_vec(k_skip)               # [6, 2]
        r_outer = transcript.sample_f128_vec(m - k_skip - N_INNER)  # [m-13, 2]
        # r = r_skip ++ small ++ medium ++ r_outer
        r = np.zeros((m, 2), dtype=np.uint64)
        r[:k_skip] = r_skip
        for i, v in enumerate(small_challenges()):
            r[k_skip + i] = _to_lohi(v)
        for i, v in enumerate(medium_challenges()):
            r[k_skip + 3 + i] = _to_lohi(v)
        if m - k_skip - N_INNER > 0:
            r[k_skip + N_INNER:] = r_outer
        return replace(carry, r=r), transcript, None


class _UrmRound(Round):
    """Round-1 univariate-skip URM (== wire round1_ab/round1_c): F8-NTT extend +
    a·b + φ8-accumulate on the GPU, then the c-claim interpolation at z. Message
    = (round1_ab, round1_c)."""

    def __init__(self, m: int, k_skip: int):
        self._m, self._k_skip = m, k_skip

    def __call__(self, carry, transcript):
        m, k_skip = self._m, self._k_skip
        # Transfer the witness to device ONCE (round1 reads a/b/c; the multilinear
        # fold_at_z reuses a/b without re-sending) — the device-resident pattern.
        a_rows = gf8.witness_to_rows(carry.a_bits, m, k_skip)
        b_rows = gf8.witness_to_rows(carry.b_bits, m, k_skip)
        c_rows = gf8.witness_to_rows(carry.c_bits, m, k_skip)
        round1_ab, round1_c = gf8.round1_rows(a_rows, b_rows, c_rows, m, k_skip, carry.r)
        transcript.observe_f128_slice(round1_ab)
        transcript.observe_f128_slice(round1_c)
        z = transcript.sample_f128()
        z_int = _to_int(z)
        # c-claim: interpolate round1_c at z.
        round1_c_int = [_to_int(round1_c[i]) for i in range(round1_c.shape[0])]
        final_c_eval = _to_lohi(_interpolate_at_z_on_lambda(round1_c_int, k_skip, z_int))
        carry = replace(carry, a_rows=a_rows, b_rows=b_rows, round1_ab=round1_ab,
                        round1_c=round1_c, z=z, final_c_eval=final_c_eval)
        return carry, transcript, (round1_ab, round1_c)


class _MultilinearRound(Round):
    """The multilinear sumcheck over the m − k_skip outer variables: fold the
    witness at z, then bind each remaining variable (round message + fold),
    finishing at ρ_last. Message = (rounds, final_a_eval, final_b_eval)."""

    def __init__(self, m: int, k_skip: int):
        self._m, self._k_skip = m, k_skip

    def __call__(self, carry, transcript):
        m, k_skip = self._m, self._k_skip
        n_mlv = m - k_skip
        r = carry.r
        z_int = _to_int(carry.z)
        # Per-round field ops are jitted (values identical → byte-match preserved)
        # so each round runs as ONE fused kernel; the module-level cache compiles
        # once and reuses across proofs.
        _round, _fold = _jit_round_fold()

        # round 2: fold witness at z + first multilinear message.
        weights = _lagrange_weights(k_skip, z_int, 0)  # S-domain
        a_mlv = jnp.asarray(_fold_at_z_rows(carry.a_rows, weights))
        b_mlv = jnp.asarray(_fold_at_z_rows(carry.b_rows, weights))
        mlv_arg = np.concatenate([_ONE[None, :], r[k_skip + 1:m]], axis=0)  # [n_mlv, 2]
        msg1, msginf = _round(a_mlv, b_mlv, jnp.asarray(mlv_arg))
        rounds = [(np.asarray(msg1), np.asarray(msginf))]
        transcript.observe_f128(rounds[0][0])
        transcript.observe_f128(rounds[0][1])
        rhos = [transcript.sample_f128()]

        # rounds 3..(n_mlv+1): fold at ρ_prev, then next message.
        for i in range(n_mlv - 1):
            r_next = np.concatenate([_ONE[None, :], r[k_skip + i + 2:m]], axis=0)
            a_mlv, b_mlv = _fold(a_mlv, b_mlv, jnp.asarray(rhos[i]))
            m1, mi = _round(a_mlv, b_mlv, jnp.asarray(r_next))
            rounds.append((np.asarray(m1), np.asarray(mi)))
            transcript.observe_f128(rounds[-1][0])
            transcript.observe_f128(rounds[-1][1])
            rhos.append(transcript.sample_f128())

        # final binding at ρ_last.
        a_mlv, b_mlv = _fold(a_mlv, b_mlv, jnp.asarray(rhos[-1]))
        final_a_eval = np.asarray(a_mlv)[0]
        final_b_eval = np.asarray(b_mlv)[0]
        transcript.observe_f128(final_a_eval)
        transcript.observe_f128(final_b_eval)
        carry = replace(carry, multilinear_rounds=rounds, final_a_eval=final_a_eval,
                        final_b_eval=final_b_eval, mlv_challenges=np.stack(rhos))
        return carry, transcript, (rounds, final_a_eval, final_b_eval)


def zerocheck_chain(m: int, k_skip: int) -> ProveChain:
    """The zerocheck sub-chain: setup → round-1 URM → multilinear sumcheck. One
    definition for the stage wiring (cf. prover.prove_fast / sp1-zorch
    prove_shard_chain)."""
    return ProveChain([_SetupRound(m, k_skip), _UrmRound(m, k_skip),
                       _MultilinearRound(m, k_skip)])


def prove_packed(a_bits, b_bits, c_bits, m: int, domain: bytes = None, ch=None) -> dict:
    """Returns the ZerocheckProof fields + the claim's z / mlv_challenges / r_rest
    (the latter for the oracle's localization cross-checks).

    A `zerocheck_chain` of stage `Round`s (setup → URM → multilinear) threading one
    `Challenger`; pass a shared `ch` (the e2e challenger carrying commit/bind state)
    to thread Fiat-Shamir through the fused prover, else a fresh Challenger(domain)
    is made."""
    k_skip = K_SKIP
    assert m >= k_skip + N_INNER, f"m must be >= {k_skip + N_INNER}"
    if ch is None:
        ch = Challenger(domain)
    carry, _ch, _msgs = zerocheck_chain(m, k_skip)(
        _ZerocheckCarry(a_bits, b_bits, c_bits), ch)
    return {
        "round1_ab": carry.round1_ab,
        "round1_c": carry.round1_c,
        "multilinear_rounds": carry.multilinear_rounds,
        "final_a_eval": carry.final_a_eval,
        "final_b_eval": carry.final_b_eval,
        "final_c_eval": carry.final_c_eval,
        # claim cross-checks:
        "z": carry.z,
        "mlv_challenges": carry.mlv_challenges,
        "r_rest": carry.r[k_skip:],
    }
