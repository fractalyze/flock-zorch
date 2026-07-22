# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Round-trip gate for the zerocheck verifier (no golden): `prove_packed` then
`verifier.verify` accepts and its reconstructed claim matches the prover's, and a
tampered proof is rejected with `ok=False`. Software-mul path, so it runs on CPU
under bazel."""
from __future__ import annotations

import dataclasses
import functools

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from absl.testing import absltest, parameterized  # noqa: E402

from flock_zorch import ghash, zerocheck  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.zerocheck import verifier  # noqa: E402

DOMAIN = b"flock-zc-verify-test"


def _lanes(x) -> np.ndarray:
    return np.asarray(ghash.to_lanes(x))


def _witness(m: int, seed: int):
    rng = np.random.default_rng(seed)
    nbytes = (1 << m) // 8

    def bits():
        return np.unpackbits(rng.integers(0, 256, nbytes, dtype=np.uint8), bitorder="little")

    a, b = bits(), bits()
    return a, b, a & b  # honest witness: a·b = c


@functools.cache  # one prove per m — both test methods replay the same proof
def _proof(m: int):
    a, b, c = _witness(m, m)
    return zerocheck.prove_packed(a, b, c, m, DOMAIN)


class ZerocheckVerifyTest(parameterized.TestCase):

    @parameterized.parameters(13, 14, 16)
    def test_accept_and_claim_reconstruction(self, m: int):
        proof = _proof(m)
        claim, _, ok = verifier.verify(m, proof, Challenger(DOMAIN))
        self.assertTrue(bool(ok))
        for name, got, want in (
            ("z", claim.z, proof.z),
            ("mlv_challenges", claim.mlv_challenges, proof.mlv_challenges),
            ("r_rest", claim.r_rest, proof.r_rest),
            ("a_eval", claim.a_eval, proof.final_a_eval),
            ("c_eval", claim.c_eval, proof.final_c_eval),
        ):
            np.testing.assert_array_equal(_lanes(got), _lanes(want), err_msg=name)

    @parameterized.parameters(13, 14, 16)
    def test_tamper_rejected(self, m: int):
        proof = _proof(m)
        # Corrupt the final â eval — the sumcheck identity must now fail.
        bad_a = _lanes(proof.final_a_eval).copy().reshape(2)
        bad_a[0] ^= np.uint64(1)
        _, _, ok_bad = verifier.verify(m, dataclasses.replace(proof, final_a_eval=bad_a),
                                       Challenger(DOMAIN))
        self.assertFalse(bool(ok_bad))


if __name__ == "__main__":
    absltest.main()
