"""Byte-equality of the Pallas factored-eq URM against the composite.

Run (GPU required — the kernel is Triton):
  FRX_PLATFORMS=cuda,cpu PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/urm_pallas_test.py
"""

from __future__ import annotations

import frx

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest  # noqa: E402

from flock_zorch import ghash  # noqa: E402
from flock_zorch.sumcheck import build_eq  # noqa: E402
from flock_zorch.zerocheck import _urm, _urm_pallas  # noqa: E402
from flock_zorch.zerocheck import prover as zc_prover  # noqa: E402


class UrmPallasTest(absltest.TestCase):
    def test_matches_composite_on_factored_eq(self) -> None:
        rows = 1 << 13  # 16 partials
        rng = np.random.default_rng(7)
        a = fnp.asarray(rng.integers(0, 2**63, size=rows, dtype=np.uint64))
        b = fnp.asarray(rng.integers(0, 2**63, size=rows, dtype=np.uint64))
        c = fnp.asarray(rng.integers(0, 2**63, size=rows, dtype=np.uint64))

        sg = build_eq(zc_prover._SMALL_G)
        mg = build_eq(zc_prover._MEDIUM_G)
        n_out = rows >> 7
        eo = ghash.to_ghash(
            fnp.asarray(rng.integers(0, 2**63, size=(n_out, 2), dtype=np.uint64))
        )
        inner = (mg[:, None] * sg[None, :]).reshape(-1)
        eqx = (eo[:, None] * inner[None, :]).reshape(-1)

        want = _urm._round1_partials(
            a.reshape(-1, 2), b.reshape(-1, 2), c.reshape(-1, 2), eqx, 6
        )
        # The composite's partial axis groups the same contiguous 512-row
        # ranges the kernel writes.
        want_lanes = np.asarray(ghash.to_lanes(want))

        eo_scaled = eo * (sg[0] * mg[0])
        got = _urm_pallas.round1_partials_pallas(a, b, c, ghash.to_lanes(eo_scaled))
        got_lanes = np.asarray(ghash.to_lanes(got))

        # The composite fixes n_partials and scales rows-per-partial; the
        # kernel fixes 512 rows per partial. XOR-regroup the composite's axis
        # to the kernel's granularity — field add is lane XOR, so the regroup
        # is exact.
        n_got = got_lanes.shape[0]
        group = want_lanes.shape[0] // n_got
        want_lanes = np.bitwise_xor.reduce(
            want_lanes.reshape(n_got, group, *want_lanes.shape[1:]), axis=1
        )
        self.assertEqual(got_lanes.shape, want_lanes.shape)
        mismatch = np.argwhere(got_lanes != want_lanes)
        if mismatch.size:
            i = tuple(mismatch[0])
            self.fail(
                f"first mismatch at {i}: got {got_lanes[i]:#x} "
                f"want {want_lanes[i]:#x} ({mismatch.shape[0]} total)"
            )

    def test_round1_rows_dispatch_matches_composite(self) -> None:
        """round1_rows takes the kernel path on packed GPU input with the
        pinned inner challenges, and its (P^AB, P^C) byte-match the
        composite core's."""
        m, k_skip = 15, 6
        rng = np.random.default_rng(11)

        def packed():
            return fnp.asarray(
                rng.integers(0, 2**63, size=(1 << (m - 7), 2), dtype=np.uint64)
            )

        a, b, c = packed(), packed(), packed()
        r_skip = ghash.to_ghash(
            fnp.asarray(rng.integers(0, 2**63, size=(k_skip, 2), dtype=np.uint64))
        )
        r_outer = ghash.to_ghash(
            fnp.asarray(
                rng.integers(0, 2**63, size=(m - k_skip - 7, 2), dtype=np.uint64)
            )
        )
        r = fnp.concatenate([r_skip, zc_prover._SMALL_G, zc_prover._MEDIUM_G, r_outer])

        self.assertTrue(_urm._round1_pallas_ok(a, b, c, m, k_skip, r))
        got = _urm.round1_rows(a, b, c, m, k_skip, r)
        want = _urm._round1_core(a, b, c, k_skip, r)
        for name, g, w in zip(("ab", "c"), got, want):
            g_lanes = np.asarray(ghash.to_lanes(g))
            w_lanes = np.asarray(ghash.to_lanes(w))
            np.testing.assert_array_equal(g_lanes, w_lanes, err_msg=name)


if __name__ == "__main__":
    absltest.main()
