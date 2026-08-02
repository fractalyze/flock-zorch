# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Native gate (no golden) for the identity-C shortcut (#192): round-1 C derived
from the lincheck stripe must reproduce the row-major drain bit-for-bit.

C is the identity (Cz = z), so the round-1 C message and the lincheck stripe are
both partial folds of the SAME witness z. Folding the stripe at r_outer and then
its middle bits is algebraically identical to the drain by the tensor
factorization `eq(r[k_skip:]) = eq(r[k_skip:k_log]) ⊗ eq(r[k_log:])`. The
full-proof byte gate (`e2e_ligerito_oracle_test`) pins this against flock at the
production shape; this pins the two producers against each other cheaply and on
CPU, including a padded (non-honest tail) witness. Software-mul path — CPU under
bazel."""
from __future__ import annotations

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
from absl.testing import absltest, parameterized  # noqa: E402

from flock_zorch import ghash, zerocheck  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.pcs.pack import pack_witness, pack_z_lincheck_from_packed  # noqa: E402
from flock_zorch.zerocheck import _urm  # noqa: E402

K_SKIP = zerocheck.prover.K_SKIP  # 6


def _lanes(x) -> np.ndarray:
    return np.asarray(ghash.to_lanes(x)).reshape(-1, 2)


def _witness(m: int, k_log: int, useful_bits: int | None, seed: int):
    """Random flat bits → (z_packed, stripe bytes). `useful_bits < 2^k_log`
    zeroes each k_log-block's tail (flock's honest padding)."""
    rng = np.random.default_rng(seed)
    z_bits = rng.integers(0, 2, size=(1 << m), dtype=np.uint8)
    k = 1 << k_log
    if useful_bits is not None and useful_bits < k:
        grid = z_bits.reshape(-1, k)
        grid[:, useful_bits:] = 0
        z_bits = grid.reshape(-1)
    z_packed = pack_witness(z_bits, m)
    stripe = pack_z_lincheck_from_packed(z_packed, m, k_log)
    return fnp.asarray(z_packed), stripe


def _r(m: int, seed: int):
    """A full-length [m] ghash challenge vector (independent of the transcript;
    the drain and the stripe fold read the SAME r)."""
    rng = np.random.default_rng(seed)
    return ghash.to_ghash(
        fnp.asarray(rng.integers(0, 2**64, size=(m, 2), dtype=np.uint64))
    )


class StripeCTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("m13_klog7_honest", 13, 7, None),
        ("m13_klog9_honest", 13, 9, None),
        ("m14_klog9_honest", 14, 9, None),
        ("m13_klog7_padded", 13, 7, 100),
        ("m14_klog9_padded", 14, 9, 401),
    )
    def test_round1_c_stripe_matches_drain(self, m, k_log, useful_bits) -> None:
        z_packed, stripe = _witness(m, k_log, useful_bits, seed=0xC57A)
        r = _r(m, seed=7)

        rows = _urm.witness_to_rows(z_packed, m, K_SKIP)
        drain_ab, drain_c = _urm.round1_rows(rows, rows, rows, m, K_SKIP, r)
        got_ab = _urm.round1_ab_rows(rows, rows, K_SKIP, r)
        got_c = _urm.round1_c_from_stripe(stripe, m, k_log, K_SKIP, r)

        np.testing.assert_array_equal(
            _lanes(got_ab), _lanes(drain_ab), err_msg="P^AB (AB-only) != drain"
        )
        np.testing.assert_array_equal(
            _lanes(got_c), _lanes(drain_c), err_msg="P^C (stripe) != drain"
        )

    def test_prove_packed_proof_byte_identical(self) -> None:
        """End-to-end: the stripe-C `prove_packed` emits the SAME wire proof and
        claim as the drain path (same domain, same challenger schedule)."""
        m, k_log = 13, 7
        z_packed, stripe = _witness(m, k_log, None, seed=0x1D_C5)
        domain = b"flock-stripe-c-test"

        p_old, c_old = zerocheck.prove_packed(
            z_packed, z_packed, z_packed, m, ch=Challenger(domain)
        )
        p_new, c_new = zerocheck.prove_packed(
            z_packed,
            z_packed,
            z_packed,
            m,
            ch=Challenger(domain),
            c_stripe=stripe,
            k_log=k_log,
        )
        np.testing.assert_array_equal(_lanes(p_new.round1_c), _lanes(p_old.round1_c))
        np.testing.assert_array_equal(_lanes(p_new.round1_ab), _lanes(p_old.round1_ab))
        np.testing.assert_array_equal(
            _lanes(p_new.final_c_eval), _lanes(p_old.final_c_eval)
        )
        np.testing.assert_array_equal(_lanes(c_new.c_eval), _lanes(c_old.c_eval))


if __name__ == "__main__":
    absltest.main()
