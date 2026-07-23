# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Round-trip gate for the lincheck verifier (no golden): prove zerocheck then
lincheck on one challenger, verify both on a fresh one, and check the lincheck
claim matches and a tampered z_partial is rejected. Software-mul → runs on CPU."""
from __future__ import annotations

import functools

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from absl.testing import absltest, parameterized  # noqa: E402

from flock_zorch import ghash, lincheck, zerocheck  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.lincheck import verifier as lcv  # noqa: E402
from flock_zorch.lincheck.prover import AbClaimPoint  # noqa: E402
from flock_zorch.pcs.pack import pack_witness, pack_z_lincheck_from_packed  # noqa: E402
from flock_zorch.zerocheck import verifier as zcv  # noqa: E402

DOMAIN = b"flock-lc-verify-test"


def _lanes(x) -> np.ndarray:
    return np.asarray(ghash.to_lanes(x))


@functools.cache  # one zerocheck+lincheck prove per config — both tests replay it
def _prove(m: int, k_log: int, k_skip: int):
    z_bits = np.random.default_rng(m).integers(0, 2, 1 << m).astype(np.uint8)
    z_packed = pack_witness(z_bits, m)
    z_lincheck = pack_z_lincheck_from_packed(z_packed, m, k_log)
    eye = np.eye(1 << k_log, dtype=np.uint64)

    chp = Challenger(DOMAIN)
    zc = zerocheck.prove_packed(z_bits, z_bits, z_bits, m, ch=chp)
    x_ab = AbClaimPoint.from_zerocheck(zc, k_log - k_skip)
    lp = lincheck.prove(z_lincheck, eye, eye, x_ab, m, k_log, k_skip, ch=chp)
    return zc, x_ab, lp, eye


def _verify(m, k_log, k_skip, zc, x_ab, lp, eye):
    chv = Challenger(DOMAIN)
    zcv.verify(m, zc, chv)
    return lcv.verify(m, k_log, k_skip, eye, eye, x_ab,
                      zc.final_a_eval, zc.final_b_eval, lp, chv)


class LincheckVerifyTest(parameterized.TestCase):

    @parameterized.parameters((13, 8, 6), (14, 9, 6))
    def test_accept_and_claim_reconstruction(self, m: int, k_log: int, k_skip: int):
        zc, x_ab, lp, eye = _prove(m, k_log, k_skip)
        claim, _, ok = _verify(m, k_log, k_skip, zc, x_ab, lp, eye)
        self.assertTrue(bool(ok))
        for name, got, want in (
            ("w", claim.w, lp.claim.w),
            ("r_inner_skip", claim.r_inner_skip, lp.claim.r_inner_skip),
            ("r_inner_rest", claim.r_inner_rest, lp.claim.r_inner_rest),
        ):
            np.testing.assert_array_equal(_lanes(got), _lanes(want), err_msg=name)

    @parameterized.parameters((13, 8, 6), (14, 9, 6))
    def test_tampered_z_partial_rejected(self, m: int, k_log: int, k_skip: int):
        zc, x_ab, lp, eye = _prove(m, k_log, k_skip)
        zp = _lanes(lp.z_partial).copy()
        zp[0, 0] ^= np.uint64(1)
        bad = lp._replace(z_partial=ghash.to_ghash(frx.numpy.asarray(zp)))
        _, _, ok_bad = _verify(m, k_log, k_skip, zc, x_ab, bad, eye)
        self.assertFalse(bool(ok_bad))


if __name__ == "__main__":
    absltest.main()
