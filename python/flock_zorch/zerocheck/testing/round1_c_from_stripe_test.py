# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Round-1 C from the lincheck stripe (no golden): the folded-C vector the
round-1 URM feeds into `_extend_folded_c` can be derived from the lincheck
stripe instead of draining the full 2^m C witness.

Since C is the identity (`Cz = z`), the same witness backs both the zerocheck C
track and the lincheck stripe. flock-zorch drains C at `r[k_skip:]` in one stage
(`_round1_partial`: `folded_c[s] = Σ_x eq(r[k_skip:], x)·c_x[s]`, a 2^m-scale
map-reduce — the #179 arena). That factors through the stripe fold at `r[k_log:]`
(`partial_fold_packed_z`) followed by a small middle fold at `r[k_skip:k_log]`:

    folded_c[s] = Σ_mid eq(r[k_skip:k_log])[mid] · z_vec[mid·2^k_skip + s]
    z_vec       = partial_fold_packed_z(stripe(z), m, k_log, r[k_log:])

so `round1_c = _extend_folded_c(folded_c)` is byte-identical either way and the
separate C drain is redundant (issue #192). Software-mul → runs on CPU.
"""
from __future__ import annotations

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

from absl.testing import absltest, parameterized  # noqa: E402

from flock_zorch import ghash, lincheck  # noqa: E402
from flock_zorch.sumcheck import build_eq  # noqa: E402

fnp = frx.numpy
_GHASH = fnp.binary_field_ghash


def _lanes(x) -> np.ndarray:
    return np.asarray(ghash.to_lanes(x))


def _rand_ghash(rng, n):
    lanes = rng.integers(0, 1 << 63, size=(n, 2), dtype=np.uint64)
    return ghash.to_ghash(fnp.asarray(lanes))


def _stripe_from_bits(z: np.ndarray, m: int, k_log: int) -> np.ndarray:
    """Pack a bit witness [2^m] into the `partial_fold_packed_z` stripe layout
    [2^(m-k_log)/8, 2^k_log], byte `[i_outer//8, i_inner]` bit b = z[(i_outer)·
    2^k_log + i_inner] with i_outer = byte·8 + b."""
    kli = 1 << k_log
    n_outer = 1 << (m - k_log)
    z2d = z.reshape(n_outer, kli).reshape(n_outer // 8, 8, kli)
    stripe = np.zeros((n_outer // 8, kli), np.uint8)
    for b in range(8):
        stripe |= (z2d[:, b, :] << b).astype(np.uint8)
    return stripe


class Round1CFromStripeTest(parameterized.TestCase):

    # need m - k_log >= 3 (n_outer = 2^(m-k_log) >= 8 to pack into bytes)
    @parameterized.parameters((14, 6, 11), (15, 6, 11), (16, 6, 11), (17, 6, 11))
    def test_folded_c_from_stripe_matches_witness_drain(self, m, k_skip, k_log):
        rng = np.random.default_rng((m << 8) | k_log)
        z = rng.integers(0, 2, size=1 << m, dtype=np.uint8)  # C = z witness
        r = _rand_ghash(rng, m)

        # Drain path (flock-zorch `_round1_partial` C branch): fold C at r[k_skip:].
        c_rows = fnp.asarray(z.reshape(1 << (m - k_skip), 1 << k_skip))
        eqx = build_eq(r[k_skip:])[:, None]
        folded_drain = fnp.sum(
            fnp.where(c_rows.astype(bool), eqx, fnp.zeros((), _GHASH)), axis=0
        )

        # Stripe path: outer fold at r[k_log:], then middle fold at r[k_skip:k_log].
        stripe = lincheck.stripe_to_device(
            _stripe_from_bits(z, m, k_log).tobytes(), m, k_log
        )
        z_vec = lincheck.partial_fold_packed_z(stripe, m, k_log, r[k_log:])
        eq_mid = build_eq(r[k_skip:k_log])[:, None]
        folded_stripe = fnp.sum(
            eq_mid * z_vec.reshape(1 << (k_log - k_skip), 1 << k_skip), axis=0
        )

        np.testing.assert_array_equal(_lanes(folded_drain), _lanes(folded_stripe))


if __name__ == "__main__":
    absltest.main()
