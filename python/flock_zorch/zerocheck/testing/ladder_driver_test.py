# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Native unit test for the multilinear ladder's two drivers (no golden).

`_ladder_on_host` picks how the ladder's rounds reach the device — one program
per pair dispatched from a host loop, or every round traced into one program —
and the pick is per backend. Neither the schedule nor the wire may depend on
that choice, and no other test can see it:

  * the full-proof zerocheck goldens run on GPU, which takes the fused driver,
    so they never exercise the host loop;
  * the CPU byte gate is PCS-level, so it does not reach the ladder at all;
  * `SquaredLadderWireTest` and `RoundPairCascadeTest` pin the ARITHMETIC, and
    pass on whichever driver is live.

So this covers the two things the split itself adds: that the drivers agree
byte for byte (which is what carries the GPU goldens' authority onto the CPU
path), and that each backend actually reaches the driver it selected —
inverting `_ladder_on_host` otherwise keeps the suite green.

Software-mul, so it runs on CPU; both drivers run on whichever backend is live,
the way `FoldArmLockstepTest` runs both fold arms.
"""

from __future__ import annotations

from unittest import mock

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

from absl.testing import absltest, parameterized  # noqa: E402

from flock_zorch import ghash, zerocheck  # noqa: E402
from flock_zorch.zerocheck import prover  # noqa: E402

DOMAIN = b"flock-zc-ladder-driver-test"


def _bits(rng, m: int) -> np.ndarray:
    return np.unpackbits(
        rng.integers(0, 256, (1 << m) // 8, dtype=np.uint8), bitorder="little"
    )


def _instance(m: int, aliased: bool):
    """(a, b, c) for one ladder shape.

    Aliased passes the SAME object three times — what routes the equal-factor
    squared ladder. The generic shape needs a, b to differ in VALUE, not just
    in identity: with equal values â and b̂ coincide, and a driver that swapped
    the two final evals — the shape of the #233 fused-program miscompile —
    would serialize identical bytes anyway. The prover does not check the
    statement, so unsatisfying random tracks are a valid input to compare two
    drivers on.
    """
    rng = np.random.default_rng(m)
    z = _bits(rng, m)
    if aliased:
        return z, z, z
    return z, _bits(rng, m), _bits(rng, m)


def _prove(a, b, c, m: int, on_host: bool):
    """One `prove_packed` with the driver forced."""
    with mock.patch.object(prover, "_ladder_on_host", return_value=on_host):
        return zerocheck.prove_packed(a, b, c, m, DOMAIN)


class LadderDriverWireTest(parameterized.TestCase):
    """Both drivers serialize the SAME proof.

    This is what lets the GPU full-proof goldens speak for the CPU path: they
    pin the fused driver against flock, and this pins the host driver against
    the fused one.
    """

    @parameterized.named_parameters(
        ("generic_m13", 13, False),
        ("generic_m14", 14, False),
        ("squared_m13", 13, True),
        ("squared_m14", 14, True),
    )
    def test_proof_bytes_identical(self, m: int, aliased: bool):
        a, b, c = _instance(m, aliased)
        host_proof, host_claim = _prove(a, b, c, m, on_host=True)
        fused_proof, fused_claim = _prove(a, b, c, m, on_host=False)

        for name in (
            "round1_ab",
            "round1_c",
            "final_a_eval",
            "final_b_eval",
            "final_c_eval",
        ):
            np.testing.assert_array_equal(
                ghash.to_lanes(getattr(host_proof, name)),
                ghash.to_lanes(getattr(fused_proof, name)),
                err_msg=name,
            )
        self.assertLen(
            host_proof.multilinear_rounds, len(fused_proof.multilinear_rounds)
        )
        for i, (host_msg, fused_msg) in enumerate(
            zip(host_proof.multilinear_rounds, fused_proof.multilinear_rounds)
        ):
            for j, which in enumerate(("G(1)", "G(inf)")):
                np.testing.assert_array_equal(
                    ghash.to_lanes(host_msg[j]),
                    ghash.to_lanes(fused_msg[j]),
                    err_msg=f"round {i} {which}",
                )
        np.testing.assert_array_equal(
            ghash.to_lanes(host_claim.mlv_challenges),
            ghash.to_lanes(fused_claim.mlv_challenges),
        )
        np.testing.assert_array_equal(
            ghash.to_lanes(host_claim.z), ghash.to_lanes(fused_claim.z)
        )


class LadderDriverSelectionTest(parameterized.TestCase):
    """The backend must reach the driver `_ladder_on_host` selected.

    Asserted at the CALL SITE, not by re-deriving the predicate: a test that
    only re-checked `_ladder_on_host` would restate the implementation and
    still pass if `_MultilinearRound` ignored it.
    """

    @parameterized.named_parameters(
        ("cpu_drives_from_the_host", "cpu", True),
        ("gpu_traces_the_whole_ladder", "gpu", False),
        # Any non-CPU backend takes the fused program — the split is "is this
        # the backend whose per-round pass outweighs its dispatch", not an
        # allowlist.
        ("tpu_traces_the_whole_ladder", "tpu", False),
    )
    def test_predicate_follows_backend(self, backend: str, on_host: bool):
        with mock.patch.object(frx, "default_backend", return_value=backend):
            self.assertEqual(prover._ladder_on_host(), on_host)

    @parameterized.named_parameters(
        ("generic_cpu", "cpu", False, True),
        ("generic_gpu", "gpu", False, False),
        ("squared_cpu", "cpu", True, True),
        ("squared_gpu", "gpu", True, False),
    )
    def test_call_site_dispatches_the_selected_driver(
        self, backend: str, aliased: bool, want_host: bool
    ):
        m = 13
        a, b, c = _instance(m, aliased)
        host_name = "_mlv_sumcheck_sq" if aliased else "_mlv_sumcheck"
        fused_name = "_MLV_SUMCHECK_SQ_FUSED" if aliased else "_MLV_SUMCHECK_FUSED"

        with (
            mock.patch.object(frx, "default_backend", return_value=backend),
            mock.patch.object(
                prover, host_name, wraps=getattr(prover, host_name)
            ) as host,
            mock.patch.object(
                prover, fused_name, wraps=getattr(prover, fused_name)
            ) as fused,
        ):
            zerocheck.prove_packed(a, b, c, m, DOMAIN)

        self.assertEqual(host.called, want_host, f"{host_name} called")
        self.assertEqual(fused.called, not want_host, f"{fused_name} called")


if __name__ == "__main__":
    absltest.main()
