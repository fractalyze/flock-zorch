# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Native unit test for `lincheck.CscCircuit.fold_alpha_batched` (no golden).

The device fold is a column-sorted prefix-XOR scan (`_csc_fold._seg_xor_fold`);
the reference here is the naive transposed binary matvec — a host numpy XOR
scatter over uint64 lanes, independent of the sort/segment/scatter plumbing —
with only the final α·a ⊕ b combine on the field dtype. This carries the
non-identity A₀/B₀ structure coverage the retired per-layer lincheck golden gate
had (the proof-level gates only drive identity or procedural circuits); the
fold's byte-identity to flock rides those proof gates."""
from __future__ import annotations

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from absl.testing import absltest, parameterized  # noqa: E402

from flock_zorch import ghash  # noqa: E402
from flock_zorch.lincheck import CscCircuit  # noqa: E402


def _rand_lanes(rng, n: int) -> np.ndarray:
    return rng.integers(0, 1 << 63, size=(n, 2), dtype=np.uint64)


def _rand_rows(rng, n_rows: int, k: int, max_nnz: int) -> list[list[int]]:
    return [
        sorted(rng.choice(k, size=rng.integers(0, max_nnz + 1), replace=False).tolist())
        for _ in range(n_rows)
    ]


def _ref_matvec_lanes(rows: list[list[int]], eq_lanes: np.ndarray, k: int) -> np.ndarray:
    """out[c] = XOR_{r: M[r,c]=1} eq[r], as a naive host scatter over lanes."""
    out = np.zeros((k, 2), np.uint64)
    for r, cols in enumerate(rows):
        for c in cols:
            out[c] ^= eq_lanes[r]
    return out


class CscFoldTest(parameterized.TestCase):

    def _assert_fold_matches(self, a_rows, b_rows, k: int, seed: int):
        rng = np.random.default_rng(seed)
        eq_lanes = _rand_lanes(rng, max(len(a_rows), len(b_rows)))
        alpha = ghash.to_ghash(_rand_lanes(rng, 1)[0])  # (2,) lanes -> ghash scalar

        got = CscCircuit(a_rows, b_rows, k).fold_alpha_batched(
            alpha, ghash.to_ghash(eq_lanes))
        want = (alpha * ghash.to_ghash(_ref_matvec_lanes(a_rows, eq_lanes, k))
                + ghash.to_ghash(_ref_matvec_lanes(b_rows, eq_lanes, k)))
        np.testing.assert_array_equal(ghash.to_lanes(got), ghash.to_lanes(want))

    @parameterized.parameters((16, 3, 0), (64, 4, 1), (256, 6, 2))
    def test_random_sparse(self, k: int, max_nnz: int, seed: int):
        rng = np.random.default_rng(seed)
        self._assert_fold_matches(
            _rand_rows(rng, k, k, max_nnz), _rand_rows(rng, k, k, max_nnz), k, seed)

    def test_skewed_column(self):
        # Every A row hits column 0 (the const_pin shape the seg-scan exists for)
        # plus one spread column; B stays random.
        k = 64
        rng = np.random.default_rng(7)
        a_rows = [[0, int(rng.integers(1, k))] for _ in range(k)]
        self._assert_fold_matches(a_rows, _rand_rows(rng, k, k, 3), k, seed=7)

    def test_empty_a_matrix(self):
        k = 32
        rng = np.random.default_rng(11)
        self._assert_fold_matches([], _rand_rows(rng, k, k, 3), k, seed=11)

    def test_empty_rows_and_boundary_column(self):
        k = 32
        a_rows = [[], [k - 1], [], [0, k - 1]] + [[] for _ in range(k - 4)]
        b_rows = [[] for _ in range(k)]
        self._assert_fold_matches(a_rows, b_rows, k, seed=13)


if __name__ == "__main__":
    absltest.main()
