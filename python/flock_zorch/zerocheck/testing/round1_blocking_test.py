# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Native unit test for round-1's row blocking (no golden).

Round-1 is a map-reduce over the row axis: `_extend_rows` transforms along the
LAST axis, so rows are independent, and the reduction collapses them to `[ell]`.
Blocking that axis is what lets m=32 — the benchmark instance — fit on a 32 GB
card at all (#179), and it is only admissible because the accumulation is `+` on
`binary_field_ghash`, i.e. XOR: associative and commutative, so partitioning the
rows cannot change the result.

"Cannot change the result" is the claim under test, and it is exact rather than
approximate, so the assertion is byte equality at several block counts. The
full-proof byte gate covers the production block size against flock; this covers
the partition argument itself, cheaply and on CPU.
"""
from __future__ import annotations

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
from absl.testing import absltest, parameterized  # noqa: E402

from flock_zorch import ghash  # noqa: E402
from flock_zorch.zerocheck import _urm  # noqa: E402

K_SKIP = 6
ELL = 1 << K_SKIP
N_ROWS = 1 << 14


def _inputs(seed: int):
    rng = np.random.default_rng(seed)
    rows = lambda: fnp.asarray(  # noqa: E731
        rng.integers(0, 2, size=(N_ROWS, ELL), dtype=np.uint8)
    )
    a, b, c = rows(), rows(), rows()
    n_challenges = K_SKIP + N_ROWS.bit_length() - 1
    r = ghash.to_ghash(
        fnp.asarray(rng.integers(0, 2**64, size=(n_challenges, 2), dtype=np.uint64))
    )
    return a, b, c, r


def _round1_at(block_rows: int, a, b, c, r):
    """`_round1_core` with the block size forced, as lanes for exact compare."""
    saved = _urm._ROUND1_BLOCK_ROWS
    try:
        _urm._ROUND1_BLOCK_ROWS = block_rows
        _urm._round1_core.clear_cache()  # block size is baked into the trace
        return [
            np.asarray(ghash.to_lanes(x)) for x in _urm._round1_core(a, b, c, K_SKIP, r)
        ]
    finally:
        _urm._ROUND1_BLOCK_ROWS = saved
        _urm._round1_core.clear_cache()


class Round1BlockingTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("2_blocks", 1 << 13),
        ("4_blocks", 1 << 12),
        ("16_blocks", 1 << 10),
        ("128_blocks", 1 << 7),
    )
    def test_blocked_matches_unblocked_exactly(self, block_rows: int) -> None:
        a, b, c, r = _inputs(seed=7)
        # A block wider than the input takes the unblocked path — the reference.
        ref = _round1_at(N_ROWS << 1, a, b, c, r)
        got = _round1_at(block_rows, a, b, c, r)
        for name, x, y in zip(("P_AB", "P_C"), ref, got):
            np.testing.assert_array_equal(
                x, y, err_msg=f"{name} differs at {N_ROWS // block_rows} blocks"
            )

    def test_production_block_size_leaves_small_instances_unblocked(self) -> None:
        # The threshold exists so an instance that already fits keeps its exact
        # program rather than becoming a scan of length 1. m<=28 is below it.
        self.assertGreaterEqual(_urm._ROUND1_BLOCK_ROWS, 1 << 22)


if __name__ == "__main__":
    absltest.main()
