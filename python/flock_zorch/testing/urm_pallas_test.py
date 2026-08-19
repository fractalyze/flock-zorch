"""Byte-equality of the Pallas factored-eq URM against the composite.

Run (GPU required — the kernel is Triton):
  FRX_PLATFORMS=cuda,cpu PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/urm_pallas_test.py
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from flock_zorch import ghash
from flock_zorch.sumcheck import build_eq
from flock_zorch.zerocheck import _urm, _urm_pallas
from flock_zorch.zerocheck import prover as zc_prover


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


if __name__ == "__main__":
    absltest.main()
