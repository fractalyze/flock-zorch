# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The CPU partial fold's table path against the portable select-XOR oracle
(no golden). The two formulations differ in everything but the answer — one
table lookup per byte vs one select-XOR step per outer bit — so this is the
gate that pins the sum tables, the stripe bit order, and the chunked loop.

Software-mul → runs on CPU."""
from __future__ import annotations

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

from absl.testing import absltest, parameterized  # noqa: E402

from flock_zorch import ghash  # noqa: E402
from flock_zorch.lincheck.prover import (  # noqa: E402
    _partial_fold_expr,
    _partial_fold_table,
    _stripe_sum_tables,
)
from flock_zorch.sumcheck import build_eq  # noqa: E402
from flock_zorch.testing._util import rand_ghash  # noqa: E402

fnp = frx.numpy


def _stripe(rng, m: int, k_log: int, density: float = 1.0):
    """A packed stripe as `partial_fold_packed_z` consumes it: `[n_outer/8, k]`
    bytes, `density` scaling how many outer bits are set."""
    n_bytes = (1 << (m - k_log)) // 8
    bits = rng.random((n_bytes, 1 << k_log, 8)) < density
    return fnp.asarray(np.packbits(bits, axis=-1, bitorder="little")[..., 0])


class StripeSumTableTest(parameterized.TestCase):

    def test_table_is_the_xor_of_the_bits_set_in_the_index(self):
        """`build_sum_table`'s contract, spelled naively over all 256 entries."""
        rng = np.random.default_rng(0)
        n_stripes = 3
        eq_outer = rand_ghash(rng, 8 * n_stripes)
        table = ghash.to_lanes(_stripe_sum_tables(eq_outer, n_stripes))
        eq8 = ghash.to_lanes(eq_outer).reshape(n_stripes, 8, 2)
        for s in range(n_stripes):
            for b in range(256):
                want = np.zeros(2, np.uint64)
                for r in range(8):
                    if b >> r & 1:
                        want ^= eq8[s, r]
                np.testing.assert_array_equal(table[s, b], want, f"stripe {s} byte {b}")


class PartialFoldTableTest(parameterized.TestCase):

    @parameterized.named_parameters(
        # (m, k_log): one stripe (chunk clamps below _STRIPES_PER_CHUNK), the
        # exact chunk, several chunks, and the m26 bench shape.
        ("one_stripe", 11, 8),
        ("one_chunk", 14, 8),
        ("many_chunks", 20, 12),
        ("bench_shape", 22, 14),
    )
    def test_matches_the_select_xor_oracle(self, m: int, k_log: int):
        rng = np.random.default_rng(m * 100 + k_log)
        zp = _stripe(rng, m, k_log)
        eq_outer = build_eq(rand_ghash(rng, m - k_log))
        n_outer = 1 << (m - k_log)
        got = _partial_fold_table(zp, eq_outer, n_outer)
        want = _partial_fold_expr(zp, eq_outer, n_outer)
        np.testing.assert_array_equal(ghash.to_lanes(got), ghash.to_lanes(want))

    @parameterized.named_parameters(("all_zero", 0.0), ("sparse", 0.05))
    def test_matches_the_oracle_on_degenerate_stripes(self, density: float):
        """A zero byte must index the field zero, or padded rows — which are
        zero in every block of an honestly padded witness — would contribute."""
        m, k_log = 20, 12
        rng = np.random.default_rng(7)
        zp = _stripe(rng, m, k_log, density)
        eq_outer = build_eq(rand_ghash(rng, m - k_log))
        n_outer = 1 << (m - k_log)
        got = _partial_fold_table(zp, eq_outer, n_outer)
        want = _partial_fold_expr(zp, eq_outer, n_outer)
        np.testing.assert_array_equal(ghash.to_lanes(got), ghash.to_lanes(want))


if __name__ == "__main__":
    absltest.main()
