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

import functools
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from flock_zorch import field, sumcheck
from flock_zorch.field import gf8
from flock_zorch.field import _to_int, _to_lohi, _int_to_ghash, _ghash_to_int
from flock_zorch.challenger import Challenger
from flock_zorch.zerocheck._fold import (
    _lagrange_weights, _interpolate_at_z_on_lambda, _fold_at_z_rows, _phi_int, _ONE,
)
from zorch.round import ProveChain, Round

K_SKIP = 6
N_INNER = 7  # 3 small + 4 medium fixed-constant inner dims
LABEL = b"flock-zerocheck-v0"


@dataclass(frozen=True)
class ZerocheckProof:
    """flock's ZerocheckProof — the round-1 URM messages, the multilinear-round
    (G(1), G(∞)) pairs, and the final a/b/c evaluations. `z`, `mlv_challenges`,
    and `r_rest` are the claim cross-checks the oracle localizes against, not wire
    fields."""

    round1_ab: Any
    round1_c: Any
    multilinear_rounds: Any
    final_a_eval: Any
    final_b_eval: Any
    final_c_eval: Any
    z: Any                # claim cross-check
    mlv_challenges: Any   # claim cross-check
    r_rest: Any           # claim cross-check


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
        out.append(_ghash_to_int(_int_to_ghash(ge) * _int_to_ghash(1 ^ ge) ** -1))
    return out


@functools.partial(jax.jit, static_argnums=(4, 5))
def _mlv_sumcheck(a_g, b_g, r_g, t, k_skip, n_mlv):
    """The whole multilinear phase as ONE device program with Fiat-Shamir inside:
    per round, the eq-weighted message pair (r[0]·G(1), G(∞)) over the shrinking
    challenge suffix, two scalar-framed observes, one squeeze, then the low-bit
    fold — finishing with the final a/b evals observed. All-ghash in-trace (the
    lane conversions happen at the caller's eager boundary): chaining lane
    bitcasts inside a trace trips the XLA simplifier mis-fold (xla#256)."""
    msgs, rhos = [], []
    for i in range(n_mlv):
        arg_g = jnp.concatenate([sumcheck.eq._ONE_G.reshape(1),
                                 r_g[k_skip + 1 + i:]], axis=0)
        m1, minf = sumcheck.round_pair_g(a_g, b_g, arg_g)
        t = t.observe_scalar(m1).observe_scalar(minf)
        t, rho = t.sample_scalar()
        msgs.append((m1, minf))
        rhos.append(rho)
        a_g = sumcheck.fold_single_g(a_g, rho)
        b_g = sumcheck.fold_single_g(b_g, rho)
    final_a, final_b = a_g[0], b_g[0]
    t = t.observe_scalar(final_a).observe_scalar(final_b)
    return t, msgs, jnp.stack(rhos), final_a, final_b


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
        z_int = _to_int(carry.z)

        # Fold the witness at z, then run the whole multilinear phase (messages,
        # Fiat-Shamir, folds, final evals) as one jitted device program. Lane <->
        # ghash conversions stay at this eager boundary (xla#256).
        weights = _lagrange_weights(k_skip, z_int, 0)  # S-domain
        a_g = field.to_ghash(jnp.asarray(_fold_at_z_rows(carry.a_rows, weights)))
        b_g = field.to_ghash(jnp.asarray(_fold_at_z_rows(carry.b_rows, weights)))
        transcript._t, msgs, rhos_g, final_a, final_b = _mlv_sumcheck(
            a_g, b_g, field.to_ghash(jnp.asarray(carry.r)), transcript._t,
            k_skip, n_mlv)

        rounds = [(field.from_ghash_host(m1), field.from_ghash_host(mi))
                  for m1, mi in msgs]
        final_a_eval = field.from_ghash_host(final_a)
        final_b_eval = field.from_ghash_host(final_b)
        carry = replace(carry, multilinear_rounds=rounds, final_a_eval=final_a_eval,
                        final_b_eval=final_b_eval,
                        mlv_challenges=field.from_ghash_host(rhos_g))
        return carry, transcript, (rounds, final_a_eval, final_b_eval)


def zerocheck_chain(m: int, k_skip: int) -> ProveChain:
    """The zerocheck sub-chain: setup → round-1 URM → multilinear sumcheck. One
    definition for the stage wiring (cf. prover.prove_fast / sp1-zorch
    prove_shard_chain)."""
    return ProveChain([_SetupRound(m, k_skip), _UrmRound(m, k_skip),
                       _MultilinearRound(m, k_skip)])


def prove_packed(a_bits, b_bits, c_bits, m: int, domain: bytes | None = None,
                 ch: Challenger | None = None) -> ZerocheckProof:
    """Returns a `ZerocheckProof` (proof fields + the claim's z / mlv_challenges /
    r_rest, the latter for the oracle's localization cross-checks).

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
    return ZerocheckProof(
        round1_ab=carry.round1_ab,
        round1_c=carry.round1_c,
        multilinear_rounds=carry.multilinear_rounds,
        final_a_eval=carry.final_a_eval,
        final_b_eval=carry.final_b_eval,
        final_c_eval=carry.final_c_eval,
        z=carry.z,
        mlv_challenges=carry.mlv_challenges,
        r_rest=carry.r[k_skip:],
    )
