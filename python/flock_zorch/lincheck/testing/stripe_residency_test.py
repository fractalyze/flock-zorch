# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Device-stripe path of `partial_fold_packed_z` (no golden): folding an
already-device stripe is bit-identical to folding the host bytes, and a
mis-shaped stripe is rejected. Software-mul → runs on CPU."""
from __future__ import annotations

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

from absl.testing import absltest, parameterized  # noqa: E402

from flock_zorch import ghash, lincheck  # noqa: E402

fnp = frx.numpy


def _lanes(x) -> np.ndarray:
    return np.asarray(ghash.to_lanes(x))


def _rand_ghash(rng, n):
    lanes = rng.integers(0, 1 << 63, size=(n, 2), dtype=np.uint64)
    return ghash.to_ghash(fnp.asarray(lanes))


class StripeResidencyTest(parameterized.TestCase):

    @parameterized.parameters((13, 8), (14, 9))
    def test_device_stripe_matches_host_bytes(self, m: int, k_log: int):
        rng = np.random.default_rng(m)
        n_bytes = (1 << (m - k_log)) // 8
        z_bytes = rng.integers(0, 256, n_bytes << k_log, dtype=np.uint8).tobytes()
        x_outer = _rand_ghash(rng, m - k_log)

        want = lincheck.partial_fold_packed_z(z_bytes, m, k_log, x_outer)
        stripe = lincheck.stripe_to_device(z_bytes, m, k_log)
        got = lincheck.partial_fold_packed_z(stripe, m, k_log, x_outer)
        np.testing.assert_array_equal(_lanes(got), _lanes(want))

    def test_mis_shaped_stripe_rejected(self):
        m, k_log = 13, 8
        rng = np.random.default_rng(0)
        x_outer = _rand_ghash(rng, m - k_log)
        wrong = fnp.asarray(np.zeros((3, 1 << k_log), np.uint8))
        with self.assertRaises(ValueError):
            lincheck.partial_fold_packed_z(wrong, m, k_log, x_outer)


if __name__ == "__main__":
    absltest.main()
