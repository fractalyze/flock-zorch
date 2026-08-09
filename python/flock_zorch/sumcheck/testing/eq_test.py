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

The chain reference runs the eager software GF(2^128) multiply, which is what
times out small CI budgets (the zorch#609 lesson), so the split case runs at
`_OUTER_SPLIT_MIN` exactly — the minimum that exercises the split — and the
per-suffix `build_eq` cross-check stays below the split.
"""
from __future__ import annotations

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
from absl.testing import absltest  # noqa: E402

from flock_zorch import ghash  # noqa: E402
from flock_zorch.sumcheck import eq  # noqa: E402


def _challenges(n: int, seed: int):
    rng = np.random.default_rng(seed)
    lanes = rng.integers(0, 1 << 63, size=(n, 2), dtype=np.uint64)
    return ghash.to_ghash(fnp.asarray(lanes))


class BuildEqSuffixTablesTest(absltest.TestCase):
    def test_matches_per_suffix_build_eq(self) -> None:
        # Below the split: every table equals an independent `build_eq` over
        # its suffix, and the trivial tail is the scalar one.
        n = 5
        cs = _challenges(n, seed=7)
        tables = eq.build_eq_suffix_tables(cs)
        self.assertLen(tables, n + 1)
        for i in range(n):
            t = tables[i]
            self.assertEqual(t.shape, (1 << (n - i),))
            self.assertTrue(bool(fnp.all(t == eq.build_eq(cs[i:]))), f"i={i}")
        self.assertTrue(bool(fnp.all(tables[n] == eq._ONE_G.reshape(1))))

    def test_outer_split_matches_chain(self) -> None:
        # At _OUTER_SPLIT_MIN the large tables are emitted as outer products
        # of a shared high half; they must stay byte-equal to the doubling
        # chain for every suffix, not just the largest.
        n = eq._OUTER_SPLIT_MIN
        cs = _challenges(n, seed=11)
        split = eq.build_eq_suffix_tables(cs)
        chain = eq._suffix_chain(cs)
        self.assertLen(split, n + 1)
        for i, (s, c) in enumerate(zip(split, chain)):
            self.assertEqual(s.shape, c.shape, f"i={i}")
            self.assertTrue(bool(fnp.all(s == c)), f"i={i}")


if __name__ == "__main__":
    absltest.main()
