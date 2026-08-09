# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Python-native gate (no golden) for the composed round pair: one
`_mlv_round_pair` program must reproduce two incumbent `_mlv_round` programs
exactly — messages, sampled challenges, folded arrays, and the threaded
transcript state — from the same starting state. The composition is a pure
reassociation of exact GF(2^128) arithmetic, so every value is equal, not
merely close; the full-proof oracle gates then pin the wire bytes."""
from __future__ import annotations

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

from absl.testing import absltest, parameterized  # noqa: E402

from flock_zorch import fs, ghash, sumcheck  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.testing._util import rand_ghash  # noqa: E402
from flock_zorch.zerocheck.prover import _mlv_round, _mlv_round_pair  # noqa: E402

DOMAIN = b"flock-zc-pair-test"


class RoundPairCascadeTest(parameterized.TestCase):

    @parameterized.parameters(2, 5, 9)
    def test_pair_matches_two_singles(self, log_n: int):
        n = 1 << log_n
        a = rand_ghash(np.random.default_rng(2 * log_n), n)
        b = rand_ghash(np.random.default_rng(2 * log_n + 1), n)
        # eq challenges, suffix layout
        cs = rand_ghash(np.random.default_rng(3 * log_n), log_n - 1)
        eq_tables = sumcheck.build_eq_suffix_tables(cs)
        one = sumcheck.eq._ONE_G

        t0 = Challenger(DOMAIN)._t
        a1, b1, t1, m1_a, minf_a, rho_a = _mlv_round(a, b, eq_tables[0], one, t0)
        a2, b2, t2, m1_b, minf_b, rho_b = _mlv_round(a1, b1, eq_tables[1], one, t1)

        tp0 = Challenger(DOMAIN)._t
        ap, bp, tp, pair_msgs, pair_rhos = _mlv_round_pair(
            a, b, eq_tables[0], eq_tables[1], one, tp0
        )
        (p1_a, pinf_a), (p1_b, pinf_b) = pair_msgs
        prho_a, prho_b = pair_rhos

        for name, got, want in (
            ("m1[i]", p1_a, m1_a),
            ("minf[i]", pinf_a, minf_a),
            ("m1[i+1]", p1_b, m1_b),
            ("minf[i+1]", pinf_b, minf_b),
            ("rho[i]", prho_a, rho_a),
            ("rho[i+1]", prho_b, rho_b),
            ("a_folded", ap, a2),
            ("b_folded", bp, b2),
        ):
            np.testing.assert_array_equal(
                ghash.to_lanes(got), ghash.to_lanes(want), err_msg=name
            )

        # The threaded transcript states agree, so every LATER draw agrees too:
        # squeeze one more challenge from each and compare.
        _, probe_pair = fs.sample_scalar(tp)
        _, probe_seq = fs.sample_scalar(t2)
        np.testing.assert_array_equal(
            ghash.to_lanes(probe_pair), ghash.to_lanes(probe_seq)
        )


if __name__ == "__main__":
    absltest.main()
