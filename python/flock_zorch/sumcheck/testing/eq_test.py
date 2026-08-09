# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Native unit test for `build_eq_suffix_tables` (no golden).

Two exact claims, asserted as byte equality on `binary_field_ghash`:

* The suffix family means what it says: `T_i == build_eq(cs[i:])` for every
  suffix — the semantic contract every consumer (the ML round loop, the
  squared-ladder weighting) relies on.
* Past `_OUTER_SPLIT_MIN` the outer-product emission stays exactly equal to
  the pure doubling chain. GF mul is exact, associative, commutative, so the
  re-association cannot change a byte — the same argument zorch#609 pins for
  `expand_eq_to_hypercube`.

The split case runs at `_OUTER_SPLIT_MIN` exactly — the minimum that exercises
the split — and the per-suffix `build_eq` cross-check stays below it.
"""
from __future__ import annotations

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

from absl.testing import absltest  # noqa: E402

from flock_zorch import ghash  # noqa: E402
from flock_zorch.sumcheck import eq  # noqa: E402
from flock_zorch.testing._util import rand_ghash  # noqa: E402


class BuildEqSuffixTablesTest(absltest.TestCase):
    def test_matches_per_suffix_build_eq(self) -> None:
        # Below the split: every table — the empty-suffix scalar tail included —
        # equals an independent `build_eq` over its suffix.
        n = 5
        cs = rand_ghash(np.random.default_rng(7), n)
        tables = eq.build_eq_suffix_tables(cs)
        self.assertLen(tables, n + 1)
        for i in range(n + 1):
            self.assertEqual(tables[i].shape, (1 << (n - i),))
            np.testing.assert_array_equal(
                ghash.to_lanes(tables[i]),
                ghash.to_lanes(eq.build_eq(cs[i:])),
                err_msg=f"i={i}",
            )

    def test_outer_split_matches_chain(self) -> None:
        # At _OUTER_SPLIT_MIN the large tables are emitted as outer products
        # of a shared high half; they must stay byte-equal to the doubling
        # chain for every suffix, not just the largest.
        n = eq._OUTER_SPLIT_MIN
        cs = rand_ghash(np.random.default_rng(11), n)
        split = eq.build_eq_suffix_tables(cs)
        chain = eq._suffix_chain(cs)
        self.assertLen(split, n + 1)
        for i, (s, c) in enumerate(zip(split, chain)):
            np.testing.assert_array_equal(
                ghash.to_lanes(s), ghash.to_lanes(c), err_msg=f"i={i}"
            )


if __name__ == "__main__":
    absltest.main()
