"""flock's zerocheck `prove_packed` — the first FULL PIOP sub-protocol with a
serializable proof — authored as a host round loop, byte-identical to flock-core's
`zerocheck::prove_packed_padded_inner`.

Proves `a(y)·b(y) ⊕ c(y) = 0 ∀ y ∈ {0,1}^m`. Structure: one univariate-skip
round-1 (URM, `_urm.round1_rows`) over K_SKIP=6 skip variables, then a multilinear
sumcheck over the remaining `m − K_SKIP` variables (the iter-10 `sumcheck`
primitives). Fiat-Shamir is the host SHA-256 `Challenger`; the bulk field arith
(`round_pair_eq`, the multilinear fold) runs on the native
`binary_field_ghash` multiply (→ clmad on GPU).

The protocol fixes the inner 7 of the `r` challenge coordinates to constants
(`small`/`medium`), and the C track is pinned at round 1 (extract_c), so only AB
participate in the multilinear rounds — `final_c_eval` is an interpolation of
`round1_c` at the URM fold-point `z`. Requires `jax_enable_x64` and `zorch` on
PYTHONPATH.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from zorch.round import prove_rounds
from zorch.stage import ProveResult, ProverStage
from zorch.sumcheck.domain import fold

from flock_zorch import ghash, sumcheck
from flock_zorch.challenger import Challenger
from flock_zorch.ghash import _ghash_to_lanes, _lanes_to_ghash
from flock_zorch.types import R1csClaim, R1csWitness, ZerocheckClaim, ZerocheckProof
from flock_zorch.zerocheck import _urm
from flock_zorch.zerocheck._fold import (
    _fold_at_z,
    _interpolate_at_z_on_lambda,
    _lagrange_weights,
)

K_SKIP = 6
N_INNER = 7  # 3 small + 4 medium fixed-constant inner dims
LABEL = b"flock-zerocheck-v0"


@dataclass(frozen=True)
class _ZerocheckCarry:
    """State threaded between the zerocheck's steps — only what a later step
    reads from an earlier one. Static config (m, k_skip) lives on the step
    instances; the `None` fields are per-step outputs set via `replace`. Not
    pytree-registered: the sequence never crosses a @jit boundary (the per-round
    field ops jit internally via _jit_round_fold)."""

    a_bits: Any
    b_bits: Any
    c_bits: Any
    r: Any = None  # ← sample_challenge_coords, seeded by prove_packed
    a_rows: Any = None  # ← _UrmRound (reused by _MultilinearRound)
    b_rows: Any = None  # ← _UrmRound
    round1_ab: Any = None  # ← _UrmRound
    round1_c: Any = None  # ← _UrmRound
    z: Any = None  # ← _UrmRound
    final_c_eval: Any = None  # ← _UrmRound
    multilinear_rounds: Any = None  # ← _MultilinearRound
    final_a_eval: Any = None  # ← _MultilinearRound
    final_b_eval: Any = None  # ← _MultilinearRound
    mlv_challenges: Any = None  # ← _MultilinearRound


def small_challenges() -> np.ndarray:
    """[φ₈(0xF7), φ₈(0x53), φ₈(0xB5)] (flock `small_challenges_ghash`).

    uint64 [3, 2].
    """
    return _urm.PHI_8_TABLE[[0xF7, 0x53, 0xB5]]


def medium_challenges() -> np.ndarray:
    """[γ^E·(1+γ^E)⁻¹ for E∈{1,2,4,8}], γ^E = single bit at lo position E
    (flock `medium_challenges_ghash`). uint64 [4, 2].

    Inverted scalar-wise: `** -1` on a ghash *array* takes numpy's ufunc path,
    which rejects the -1 exponent; zk_dtypes' host inverse is scalar-only."""
    gamma = _lanes_to_ghash(np.array([[1 << e, 0] for e in (1, 2, 4, 8)], np.uint64))
    one_plus = _lanes_to_ghash(
        np.array([[1 ^ (1 << e), 0] for e in (1, 2, 4, 8)], np.uint64)
    )
    return _ghash_to_lanes(
        np.array([g * gp1**-1 for g, gp1 in zip(gamma, one_plus)], ghash._GHASH_HOST)
    )


_SMALL_G = ghash.to_ghash(
    fnp.asarray(small_challenges())
)  # [3] ghash — fixed inner challenges
_MEDIUM_G = ghash.to_ghash(fnp.asarray(medium_challenges()))  # [4] ghash


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
    return fold(a_g, rho, msb=False), fold(b_g, rho, msb=False), t, m1, minf, rho


@frx.jit
def _mlv_round_pair(a_g, b_g, eq_g, eq_next_g, r0_g, t):
    """TWO multilinear rounds as one traced block — the cascade composition.

    Round i's message reads the array as usual; round i+1's message is CARRIED
    as a deferred quadratic in the not-yet-sampled ρ_i
    (`sumcheck.round_pair_eq_deferred`), so it needs no second array read, and
    the two folds compose into one quartering pass. Pure reassociation of
    exact GF(2^128) arithmetic — the transcript bytes cannot move — and per
    round pair it deletes one full array read (the second message pass) and
    one intermediate half-size write+read (the second fold's input)."""
    m1, minf = sumcheck.round_pair_eq(a_g, b_g, eq_g, r0_g)
    g_one, g_inf = sumcheck.round_pair_eq_deferred(a_g, b_g, eq_next_g)

    t = t.observe_scalar(m1).observe_scalar(minf)
    t, rho = t.sample_scalar()

    m1_next = r0_g * sumcheck.eval_deferred(g_one, rho)
    minf_next = sumcheck.eval_deferred(g_inf, rho)
    t = t.observe_scalar(m1_next).observe_scalar(minf_next)
    t, rho_next = t.sample_scalar()

    a2 = fold(fold(a_g, rho, msb=False), rho_next, msb=False)
    b2 = fold(fold(b_g, rho, msb=False), rho_next, msb=False)
    return a2, b2, t, ((m1, minf), (m1_next, minf_next)), (rho, rho_next)


@frx.jit
def _mlv_sumcheck(a_g, b_g, eq_tables, r0_g, t):
    """Run the complete multilinear tail as one device program.

    The round shapes shrink statically, so tracing the tuple of suffix tables
    unrolls the protocol without a host-controlled loop.  This used to be a bad
    trade while each SHA-256 transcript hop lowered as a streaming loop; the
    dedicated SHA kernels and current XLA fusion make it worth re-evaluating.

    Rounds trace in composed pairs (`_mlv_round_pair`); the pair's deferred
    group of four holds exactly while two rounds remain, so an odd round
    count leaves one single `_mlv_round` at the end.
    """
    rounds, rhos = [], []
    n_mlv = len(eq_tables)
    for i in range(0, n_mlv - 1, 2):
        a_g, b_g, t, msgs, pair_rhos = _mlv_round_pair(
            a_g, b_g, eq_tables[i], eq_tables[i + 1], r0_g, t
        )
        rounds.extend(msgs)
        rhos.extend(pair_rhos)
    if n_mlv % 2:
        a_g, b_g, t, m1, minf, rho = _mlv_round(a_g, b_g, eq_tables[-1], r0_g, t)
        rounds.append((m1, minf))
        rhos.append(rho)
    final_a, final_b = a_g[0], b_g[0]
    t = t.observe_scalar(final_a).observe_scalar(final_b)
    return t, tuple(rounds), fnp.stack(rhos), final_a, final_b


@frx.jit
def _mlv_round_sq(a_g, eq_sqrt_g, r0_g, t):
    """`_mlv_round` for the equal-factor ladder (b ≡ a): the squared-sum message
    (`sumcheck.round_pair_eq_sq`, 2 muls/pair against 4) over a √eq suffix table,
    and ONE fold — the two factor states are the same vector, folded at the same
    ρ, so they stay the same vector."""
    m1, minf = sumcheck.round_pair_eq_sq(a_g, eq_sqrt_g, r0_g)
    t = t.observe_scalar(m1).observe_scalar(minf)
    t, rho = t.sample_scalar()
    return fold(a_g, rho, msb=False), t, m1, minf, rho


@frx.jit
def _mlv_round_pair_sq(a_g, eq_sqrt_next_g, sqrt_c_g, r0_g, t):
    """`_mlv_round_pair` for the equal-factor ladder (b ≡ a), on the four-sum
    basis: all six pair scalars come from `sumcheck.round_pair_eq_sq_basis`'s
    base sums over round i+1's SMALLER table (√c_i beside it) — round i's own
    table is never read. The two folds compose into one quartering pass of
    the single factor state."""
    m1, minf, g_one, g_inf = sumcheck.round_pair_eq_sq_basis(
        a_g, eq_sqrt_next_g, sqrt_c_g, r0_g
    )

    t = t.observe_scalar(m1).observe_scalar(minf)
    t, rho = t.sample_scalar()

    m1_next = r0_g * sumcheck.eval_deferred_sq(g_one, rho)
    minf_next = sumcheck.eval_deferred_sq(g_inf, rho)
    t = t.observe_scalar(m1_next).observe_scalar(minf_next)
    t, rho_next = t.sample_scalar()

    a2 = fold(fold(a_g, rho, msb=False), rho_next, msb=False)
    return a2, t, ((m1, minf), (m1_next, minf_next)), (rho, rho_next)


@frx.jit
def _mlv_sumcheck_sq(a_g, eq_sqrt_tables, cs_sqrt_g, r0_g, t):
    """`_mlv_sumcheck` on the equal-factor path — the same wire: message values
    are exact field identities of the generic pair's, and the final b̂ observe
    repeats â's value, which is what the generic path serializes too.

    Rounds trace in composed pairs (`_mlv_round_pair_sq`) with the odd tail on
    the single round — the same schedule as the generic ladder. A pair reads
    ONLY round i+1's table plus √c_i (= `cs_sqrt_g[i]`, the challenge its
    table chain absorbed); slots outside `_sq_pair_reads` may arrive as None.
    The asserts fail at trace time, next to the schedule, if the emission
    keep ever drifts from the reads."""
    rounds, rhos = [], []
    n_mlv = len(eq_sqrt_tables)
    for i in range(0, n_mlv - 1, 2):
        assert eq_sqrt_tables[i + 1] is not None, f"pair table T_{i + 1} not emitted"
        a_g, t, msgs, pair_rhos = _mlv_round_pair_sq(
            a_g, eq_sqrt_tables[i + 1], cs_sqrt_g[i], r0_g, t
        )
        rounds.extend(msgs)
        rhos.extend(pair_rhos)
    if n_mlv % 2:
        assert eq_sqrt_tables[-1] is not None, "tail table not emitted"
        a_g, t, m1, minf, rho = _mlv_round_sq(a_g, eq_sqrt_tables[-1], r0_g, t)
        rounds.append((m1, minf))
        rhos.append(rho)
    final_a = a_g[0]
    t = t.observe_scalar(final_a).observe_scalar(final_a)
    return t, tuple(rounds), fnp.stack(rhos), final_a


@frx.jit
def _observe_finals(t, final_a, final_b):
    return t.observe_scalar(final_a).observe_scalar(final_b)


_EQ_TABLES = frx.jit(sumcheck.build_eq_suffix_tables)
_SQRT = frx.jit(sumcheck.sqrt_ghash)


def _sq_pair_reads(n_mlv):
    """The sq pair schedule's table read set: round i+1's table per pair (the
    odd indices) plus the tail round's own T_{n_mlv-1}. The single source of
    truth — `_sq_pair_suffix_tables` emits exactly this set and
    `_mlv_sumcheck_sq` asserts it before reading. Why the rest is dead is
    `sumcheck.round_pair_eq_sq_basis`'s doubling identity."""
    return lambda i: i % 2 == 1 or i == n_mlv - 1


def _sq_pair_suffix_tables(cs_sqrt_g):
    """The √eq suffix family restricted to `_sq_pair_reads` — the dead
    tables, the largest included, are never emitted."""
    return sumcheck.build_eq_suffix_tables(
        cs_sqrt_g, keep=_sq_pair_reads(int(cs_sqrt_g.shape[0]) + 1)
    )


_EQ_TABLES_SQ = frx.jit(_sq_pair_suffix_tables)


def sample_challenge_coords(transcript, m: int, k_skip: int):
    """Draw the zerocheck's challenge coordinates: ``k_skip`` skip challenges,
    then ``m - k_skip - N_INNER`` outer ones, with the fixed inner-7 constants
    sitting between them.

    A shared function both roles call, not a step: it only advances the
    transcript and owns no proof section. Both sides need the same schedule but
    keep different slices of it — the prover the whole vector ``r``, the verifier
    only ``r[k_skip:]`` — and writing the draw once is what stops those two from
    drifting apart.
    """
    transcript.observe_label(LABEL)
    r_skip = transcript.sample_f128(k_skip)
    r_outer = transcript.sample_f128(m - k_skip - N_INNER)
    return r_skip, r_outer


class _UrmRound:
    """Round-1 univariate-skip URM (== wire round1_ab/round1_c): F8-NTT extend +
    a·b + φ8-accumulate on the GPU, then the c-claim interpolation at z. Message
    = (round1_ab, round1_c)."""

    def __init__(self, m: int, k_skip: int):
        self._m, self._k_skip = m, k_skip

    def __call__(self, carry, transcript):
        m, k_skip = self._m, self._k_skip
        # Round 1 takes the witness in whatever form it arrived: the URM
        # composite unpacks packed lanes in-fusion (`_urm._round1_input_rows`),
        # so nothing here materializes the 8x larger bit rows. The multilinear
        # fold below likewise keeps the compact form when it has one.
        round1_ab, round1_c = _urm.round1_rows(
            carry.a_bits, carry.b_bits, carry.c_bits, m, k_skip, carry.r
        )
        transcript.observe_f128(round1_ab)  # native ghash, no host round trip
        transcript.observe_f128(round1_c)
        z = transcript.sample_f128()
        # c-claim: interpolate round1_c at z.
        final_c_eval = _interpolate_at_z_on_lambda(round1_c, k_skip, z)
        a_fold = (
            carry.a_bits
            if _urm.is_packed_witness(carry.a_bits)
            else _urm.witness_to_rows(carry.a_bits, m, k_skip)
        )
        # Alias, don't recompute, when both factors are the same witness (the
        # identity-instance call a = b = ẑ): `a_rows is b_rows` is what routes
        # the multilinear tail onto its equal-factor path.
        b_fold = (
            a_fold
            if carry.b_bits is carry.a_bits
            else (
                carry.b_bits
                if _urm.is_packed_witness(carry.b_bits)
                else _urm.witness_to_rows(carry.b_bits, m, k_skip)
            )
        )
        carry = replace(
            carry,
            a_rows=a_fold,
            b_rows=b_fold,
            round1_ab=round1_ab,
            round1_c=round1_c,
            z=z,
            final_c_eval=final_c_eval,
        )
        return carry, transcript, (round1_ab, round1_c)


class _MultilinearRound:
    """The multilinear sumcheck over the m − k_skip outer variables: fold the
    witness at z, then bind each remaining variable (round message + fold),
    finishing at ρ_last. Message = (rounds, final_a_eval, final_b_eval)."""

    def __init__(self, m: int, k_skip: int):
        self._m, self._k_skip = m, k_skip

    def __call__(self, carry, transcript):
        m, k_skip = self._m, self._k_skip
        # Fold the witness at z, then run the multilinear tail as one jitted
        # device program with Fiat-Shamir inside — all on the dtype, so the lanes
        # only reappear where a proof message is serialized.
        weights = _lagrange_weights(k_skip, carry.z, 0)  # S-domain, ghash [ell]
        a_g = _fold_at_z(carry.a_rows, weights)
        r_g = carry.r

        # All rounds' eq suffix tables in one program (round i reads
        # eq(r[k_skip+1+i:])); r[0] of every round's message is fixed to one.
        # Equal factors (the identity instance a = b = ẑ, aliased by _UrmRound;
        # hash circuits carry distinct Az/Bz and stay generic) take the
        # squared-sum ladder: every product is a square and char-2 squaring
        # distributes over the XOR-sum, so the message needs √eq tables (the
        # same doubling chain over √challenges) and half the muls, and the
        # second fold disappears — and the pair schedule collapses further
        # onto the four-sum basis (sumcheck.round_pair_eq_sq_basis), which is
        # what lets _sq_pair_suffix_tables drop the tables it makes dead.
        # Byte-identical wire either way.
        if carry.b_rows is carry.a_rows:
            cs_sqrt = _SQRT(r_g[k_skip + 1 :])
            eq_tables = _EQ_TABLES_SQ(cs_sqrt)
            transcript._t, rounds, rhos, final_a = _mlv_sumcheck_sq(
                a_g, eq_tables, cs_sqrt, sumcheck.eq._ONE_G, transcript._t
            )
            final_b = final_a
        else:
            b_g = _fold_at_z(carry.b_rows, weights)
            eq_tables = _EQ_TABLES(r_g[k_skip + 1 :])
            transcript._t, rounds, rhos, final_a, final_b = _mlv_sumcheck(
                a_g, b_g, eq_tables, sumcheck.eq._ONE_G, transcript._t
            )

        final_a_eval = final_a
        final_b_eval = final_b
        carry = replace(
            carry,
            multilinear_rounds=list(rounds),
            final_a_eval=final_a_eval,
            final_b_eval=final_b_eval,
            mlv_challenges=rhos,
        )  # native ghash open-point coords
        return carry, transcript, (rounds, final_a_eval, final_b_eval)


def zerocheck_steps(m: int, k_skip: int) -> list:
    """The zerocheck sub-sequence: round-1 URM → multilinear sumcheck. One
    definition for the wiring (cf. prover.prove_fast). Drive with
    ``zorch.round.prove_rounds``, after ``sample_challenge_coords`` has drawn the
    challenge vector."""
    return [_UrmRound(m, k_skip), _MultilinearRound(m, k_skip)]


def prove_packed(
    a_bits,
    b_bits,
    c_bits,
    m: int,
    domain: bytes | None = None,
    ch: Challenger | None = None,
) -> tuple[ZerocheckProof, ZerocheckClaim]:
    """Returns the wire `ZerocheckProof` and the `ZerocheckClaim` it discharges
    into — the evaluation point and the â/b̂/ĉ evals bound at it.

    A `zerocheck_steps` sequence (URM → multilinear) threading one
    `Challenger`; pass a shared `ch` (the e2e challenger carrying commit/bind state)
    to thread Fiat-Shamir through the fused prover, else a fresh Challenger(domain)
    is made.

    Passing the SAME array object as `a_bits` and `b_bits` (the identity R1CS
    instance a = b = c = ẑ — ZerocheckProver's call) routes the multilinear
    ladder onto the equal-factor squared-sum path — the same wire bytes at half
    the GF muls (sumcheck.round_pair_eq_sq). Hash circuits pass distinct
    Az / Bz tracks and keep the generic ladder."""
    k_skip = K_SKIP
    assert m >= k_skip + N_INNER, f"m must be >= {k_skip + N_INNER}"
    if ch is None:
        assert domain is not None, "pass either a domain or a Challenger"
        ch = Challenger(domain)
    r_skip, r_outer = sample_challenge_coords(ch, m, k_skip)
    # r = r_skip ++ small ++ medium ++ r_outer, assembled on the dtype.
    r = fnp.concatenate([r_skip, _SMALL_G, _MEDIUM_G, r_outer])
    carry, _ch, _msgs = prove_rounds(
        zerocheck_steps(m, k_skip), _ZerocheckCarry(a_bits, b_bits, c_bits, r=r), ch
    )
    proof = ZerocheckProof(
        round1_ab=carry.round1_ab,
        round1_c=carry.round1_c,
        multilinear_rounds=carry.multilinear_rounds,
        final_a_eval=carry.final_a_eval,
        final_b_eval=carry.final_b_eval,
        final_c_eval=carry.final_c_eval,
    )
    claim = ZerocheckClaim(
        z=carry.z,
        mlv_challenges=carry.mlv_challenges,
        # native ghash: the c-open point coords; the byte gate lanes-converts.
        r_rest=carry.r[k_skip:],
        a_eval=carry.final_a_eval,
        b_eval=carry.final_b_eval,
        c_eval=carry.final_c_eval,
    )
    return proof, claim


class ZerocheckProver(
    ProverStage[R1csClaim, R1csWitness, ZerocheckClaim, ZerocheckProof]
):
    """Reduce the R1CS Hadamard constraint to evaluation claims on â, b̂, ĉ.

    Runs on the identity witness (a = b = c = ẑ), which is what makes the
    identity R1CS gate meaningful.
    """

    def __init__(self, m):
        self._m = m

    def prove(
        self, claim: R1csClaim, witness: R1csWitness, transcript
    ) -> ProveResult[ZerocheckClaim, ZerocheckProof]:
        proof, reduced = prove_packed(
            witness.z_packed,
            witness.z_packed,
            witness.z_packed,
            self._m,
            ch=transcript,
        )
        return ProveResult(reduced, proof, transcript)
