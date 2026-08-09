# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Native unit test for the equal-factor (squared-sum) ladder (no golden).

The identity R1CS instance proves ẑ∘ẑ ⊕ ẑ = 0 — both product factors of every
multilinear round are the same vector (hash circuits carry distinct Az/Bz and
keep the generic ladder) — and the squared-sum ladder computes each such round
message as (Σ √eq·a₁)² / (Σ √eq·S)² instead of Σ eq·a₁·b₁ / Σ eq·S·S, half the
GF(2¹²⁸) multiplies. The claim that makes this shippable is exactness:
char-2 squaring is a field automorphism, so the rearranged messages are the SAME
field elements, not approximations. The assertions are therefore byte equality:

  * `sqrt_ghash` inverts squaring on arbitrary elements (zero and one included);
  * a √challenge suffix chain IS the elementwise √ of the challenge chain;
  * `round_pair_eq_sq` == `round_pair_eq` on random equal-factor rounds;
  * `prove_packed(z, z, z)` (aliased → squared ladder) serializes byte-identically
    to `prove_packed(z, z.copy(), z)` (distinct objects → generic ladder).

The full-proof byte gates pin the production path against flock; this covers the
identity itself, cheaply and on CPU (software mul).
"""
from __future__ import annotations

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
from absl.testing import absltest, parameterized  # noqa: E402

from flock_zorch import ghash, sumcheck, zerocheck  # noqa: E402
from flock_zorch.testing._util import rand_ghash  # noqa: E402

DOMAIN = b"flock-zc-squared-ladder-test"


class SqrtGhashTest(absltest.TestCase):

    def test_squares_back(self):
        x = rand_ghash(np.random.default_rng(0), 64)
        r = sumcheck.sqrt_ghash(x)
        np.testing.assert_array_equal(ghash.to_lanes(r * r), ghash.to_lanes(x))

    def test_zero_and_one_fixed(self):
        zero_one = ghash.to_ghash(fnp.asarray([[0, 0], [1, 0]], dtype=fnp.uint64))
        np.testing.assert_array_equal(
            ghash.to_lanes(sumcheck.sqrt_ghash(zero_one)), ghash.to_lanes(zero_one)
        )


class RoundPairEqSqTest(parameterized.TestCase):

    @parameterized.parameters(1, 4, 7)
    def test_matches_generic_round(self, n: int):
        """One equal-factor round at 2^(n+1) state: sq message == generic message,
        with the √eq table built by the SAME suffix chain over √challenges."""
        rng = np.random.default_rng(n)
        a = rand_ghash(rng, 2 ** (n + 1))
        cs = rand_ghash(rng, n)
        eq = sumcheck.build_eq_suffix_tables(cs)[0]
        eq_sqrt = sumcheck.build_eq_suffix_tables(sumcheck.sqrt_ghash(cs))[0]
        # The chain commutes with √ (√ is an automorphism): √(eq table) exactly.
        np.testing.assert_array_equal(
            ghash.to_lanes(eq_sqrt * eq_sqrt), ghash.to_lanes(eq)
        )
        r0 = rand_ghash(rng, 1)[0]
        want = sumcheck.round_pair_eq(a, a, eq, r0)
        got = sumcheck.round_pair_eq_sq(a, eq_sqrt, r0)
        np.testing.assert_array_equal(ghash.to_lanes(got[0]), ghash.to_lanes(want[0]))
        np.testing.assert_array_equal(ghash.to_lanes(got[1]), ghash.to_lanes(want[1]))


class RoundPairEqSqBasisTest(parameterized.TestCase):

    @parameterized.parameters(2, 5, 8)
    def test_reproduces_all_six_pair_scalars(self, n: int):
        """The composed sq pair factors through FOUR base sums over the SMALLER
        table: the √eq chain's doubling step (√T_i[2z] = (1+√c_i)·√T_{i+1}[z],
        √T_i[2z+1] = √c_i·√T_{i+1}[z]) makes round i's message a linear
        combination of the same s_j = Σ √T_{i+1}·v_j the deferred coefficient
        pairs read — so the basis form must equal `round_pair_eq_sq` over √T_i
        AND `round_pair_eq_deferred_sq` over √T_{i+1}, byte for byte."""
        rng = np.random.default_rng(100 + n)
        a = rand_ghash(rng, 1 << (n + 1))
        sqrt_cs = sumcheck.sqrt_ghash(rand_ghash(rng, n))
        tables = sumcheck.build_eq_suffix_tables(sqrt_cs)
        r0 = rand_ghash(rng, 1)[0]

        want_m1, want_minf = sumcheck.round_pair_eq_sq(a, tables[0], r0)
        want_one, want_inf = sumcheck.round_pair_eq_deferred_sq(a, tables[1])

        m1, minf, g_one, g_inf = sumcheck.round_pair_eq_sq_basis(
            a, tables[1], sqrt_cs[0], r0
        )
        for name, got, want in (
            ("m1", m1, want_m1),
            ("minf", minf, want_minf),
            ("one_lo", g_one[0], want_one[0]),
            ("one_hi", g_one[1], want_one[1]),
            ("inf_lo", g_inf[0], want_inf[0]),
            ("inf_hi", g_inf[1], want_inf[1]),
        ):
            np.testing.assert_array_equal(
                ghash.to_lanes(got), ghash.to_lanes(want), err_msg=name
            )


class SquaredLadderWireTest(parameterized.TestCase):

    @parameterized.parameters(13, 14)
    def test_proof_bytes_unchanged(self, m: int):
        """The squared ladder (aliased booleanity witness) and the generic ladder
        (same values, distinct objects) serialize the SAME proof."""
        rng = np.random.default_rng(m)
        z = np.unpackbits(
            rng.integers(0, 256, (1 << m) // 8, dtype=np.uint8), bitorder="little"
        )
        proof_sq, claim_sq = zerocheck.prove_packed(z, z, z, m, DOMAIN)
        proof_gen, claim_gen = zerocheck.prove_packed(z, z.copy(), z, m, DOMAIN)
        for name in (
            "round1_ab",
            "round1_c",
            "final_a_eval",
            "final_b_eval",
            "final_c_eval",
        ):
            np.testing.assert_array_equal(
                ghash.to_lanes(getattr(proof_sq, name)),
                ghash.to_lanes(getattr(proof_gen, name)),
                err_msg=name,
            )
        for i, (got, want) in enumerate(
            zip(proof_sq.multilinear_rounds, proof_gen.multilinear_rounds)
        ):
            np.testing.assert_array_equal(
                ghash.to_lanes(got[0]),
                ghash.to_lanes(want[0]),
                err_msg=f"round {i} G(1)",
            )
            np.testing.assert_array_equal(
                ghash.to_lanes(got[1]),
                ghash.to_lanes(want[1]),
                err_msg=f"round {i} G(inf)",
            )
        np.testing.assert_array_equal(
            ghash.to_lanes(claim_sq.mlv_challenges),
            ghash.to_lanes(claim_gen.mlv_challenges),
        )


if __name__ == "__main__":
    absltest.main()
