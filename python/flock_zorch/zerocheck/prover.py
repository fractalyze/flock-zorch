"""flock's zerocheck `prove_packed` — the first FULL PIOP sub-protocol with a
serializable proof — authored as a host round loop, byte-identical to flock-core's
`zerocheck::prove_packed_padded_inner`.

Proves `a(y)·b(y) ⊕ c(y) = 0 ∀ y ∈ {0,1}^m`. Structure: one univariate-skip
round-1 (URM, `_urm.round1_naive`) over K_SKIP=6 skip variables, then a multilinear
sumcheck over the remaining `m − K_SKIP` variables (the iter-10 `sumcheck`
primitives). Fiat-Shamir is the host SHA-256 `Challenger`; the bulk field arith
(`round_pair`, the multilinear fold) runs on the native `binary_field_ghash` multiply (→ clmad on GPU).

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
import frx
import frx.numpy as jnp

from flock_zorch import ghash, sumcheck
from flock_zorch.zerocheck import _urm
from flock_zorch.ghash import _lanes_to_ghash, _ghash_to_lanes
from flock_zorch.challenger import Challenger
from flock_zorch.zerocheck._fold import (
    _lagrange_weights, _interpolate_at_z_on_lambda, _fold_at_z,
)
from zorch.round import ProveChain, Round
from zorch.sumcheck.domain import fold

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


def small_challenges() -> np.ndarray:
    """[φ₈(0xF7), φ₈(0x53), φ₈(0xB5)] (flock `small_challenges_ghash`). uint64 [3, 2]."""
    return _urm.PHI_8_TABLE[[0xF7, 0x53, 0xB5]]


def medium_challenges() -> np.ndarray:
    """[γ^E·(1+γ^E)⁻¹ for E∈{1,2,4,8}], γ^E = single bit at lo position E
    (flock `medium_challenges_ghash`). uint64 [4, 2].

    Inverted scalar-wise: `** -1` on a ghash *array* takes numpy's ufunc path,
    which rejects the -1 exponent; zk_dtypes' host inverse is scalar-only."""
    gamma = _lanes_to_ghash(np.array([[1 << e, 0] for e in (1, 2, 4, 8)], np.uint64))
    one_plus = _lanes_to_ghash(np.array([[1 ^ (1 << e), 0] for e in (1, 2, 4, 8)], np.uint64))
    return _ghash_to_lanes(
        np.array([g * gp1 ** -1 for g, gp1 in zip(gamma, one_plus)], ghash._GHASH_HOST))


_SMALL_G = ghash.to_ghash(jnp.asarray(small_challenges()))    # [3] ghash — fixed inner challenges
_MEDIUM_G = ghash.to_ghash(jnp.asarray(medium_challenges()))  # [4] ghash


@frx.jit
def _mlv_round(a_g, b_g, eq_g, r0_g, t):
    """ONE multilinear round as one device program: the eq-weighted message pair
    (r[0]·G(1), G(∞)) over a precomputed suffix table, two scalar-framed
    observes, one squeeze, then the low-bit fold. Per-round, not phase-unrolled:
    the unrolled n_mlv-round graph compiled for minutes and ran ~6x slower than
    its parts (XLA fusion/scheduling collapses on it). The eq table comes in as
    an operand — building it in-round re-compiled the whole doubling chain into
    every round program (~13 s × n_mlv of the cold wall). All-ghash in-trace —
    the operands arrive on the dtype and only the proof messages leave it."""
    m1, minf = sumcheck.round_pair_eq(a_g, b_g, eq_g, r0_g)
    t = t.observe_scalar(m1).observe_scalar(minf)
    t, rho = t.sample_scalar()
    return fold(a_g, rho, msb=False), fold(b_g, rho, msb=False), \
        t, m1, minf, rho


@frx.jit
def _observe_finals(t, final_a, final_b):
    return t.observe_scalar(final_a).observe_scalar(final_b)


_EQ_TABLES = frx.jit(sumcheck.build_eq_suffix_tables)


class _SetupRound(Round):
    """Sample the challenge vector r and fix the inner-7 constants (small ++
    medium). No proof message — writes r onto the carry."""

    def __init__(self, m: int, k_skip: int):
        self._m, self._k_skip = m, k_skip

    def __call__(self, carry, transcript):
        m, k_skip = self._m, self._k_skip
        transcript.observe_label(LABEL)
        # r = r_skip ++ small ++ medium ++ r_outer, assembled on the dtype.
        r_skip = transcript.sample_f128(k_skip)
        r_outer = transcript.sample_f128(m - k_skip - N_INNER)
        r = jnp.concatenate([r_skip, _SMALL_G, _MEDIUM_G, r_outer])
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
        a_rows = _urm.witness_to_rows(carry.a_bits, m, k_skip)
        b_rows = _urm.witness_to_rows(carry.b_bits, m, k_skip)
        c_rows = _urm.witness_to_rows(carry.c_bits, m, k_skip)
        round1_ab, round1_c = _urm.round1_rows(a_rows, b_rows, c_rows, m, k_skip, carry.r)
        transcript.observe_f128(ghash.to_ghash(jnp.asarray(round1_ab)))
        transcript.observe_f128(ghash.to_ghash(jnp.asarray(round1_c)))
        z = transcript.sample_f128()
        # c-claim: interpolate round1_c at z.
        final_c_eval = _interpolate_at_z_on_lambda(round1_c, k_skip, z)
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

        # Fold the witness at z, then run each multilinear round as one jitted
        # device program with Fiat-Shamir inside — all on the dtype, so the lanes
        # only reappear where a proof message is serialized.
        weights = _lagrange_weights(k_skip, carry.z, 0)  # S-domain, ghash [ell]
        a_g = _fold_at_z(carry.a_rows, weights)
        b_g = _fold_at_z(carry.b_rows, weights)
        r_g = carry.r

        t = transcript._t
        # All rounds' eq suffix tables in one program (round i reads
        # eq(r[k_skip+1+i:])); r[0] of every round's message is fixed to one.
        eq_tables = _EQ_TABLES(r_g[k_skip + 1:])
        rounds, rhos = [], []
        for i in range(n_mlv):
            a_g, b_g, t, m1, minf, rho = _mlv_round(
                a_g, b_g, eq_tables[i], sumcheck.eq._ONE_G, t)
            rounds.append((m1, minf))
            rhos.append(rho)
        final_a, final_b = a_g[0], b_g[0]
        transcript._t = _observe_finals(t, final_a, final_b)

        final_a_eval = final_a
        final_b_eval = final_b
        carry = replace(carry, multilinear_rounds=rounds, final_a_eval=final_a_eval,
                        final_b_eval=final_b_eval,
                        mlv_challenges=jnp.stack(rhos))       # native ghash open-point coords
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
        r_rest=carry.r[k_skip:],   # native ghash: the c-open point coords; byte-gate lanes-converts
    )
