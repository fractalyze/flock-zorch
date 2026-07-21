# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Pure-host gates for the witness packers, checked against the layouts their frx
consumers document — `prover._unpack_bits` (z_packed) and
`lincheck.partial_fold_packed_z` (the lincheck bytes). No golden, no field ops."""
from __future__ import annotations

import numpy as np
from absl.testing import absltest, parameterized

from flock_zorch.pcs.pack import (
    _unpack_flat,
    pack_witness,
    pack_z_lincheck_from_packed,
)


def _rand_bits(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)


class PackWitnessTest(parameterized.TestCase):
    @parameterized.parameters(7, 8, 13, 16)
    def test_pack_unpack_roundtrip(self, m: int) -> None:
        w = _rand_bits(m, 1 << m)
        self.assertTrue(np.array_equal(_unpack_flat(pack_witness(w, m)), w))

    def test_lsb_first_lo_hi_split(self) -> None:
        # element 0 covers bits [0..128): bit i → 2^i in lo (i<64) / hi (i≥64).
        w = np.zeros(128, np.uint8)
        w[0] = w[3] = w[64] = 1
        lane = pack_witness(w, 7)
        self.assertEqual(int(lane[0, 0]), 0b1001)  # lo: bits 0 and 3
        self.assertEqual(int(lane[0, 1]), 1)  # hi: bit 64 → bit 0

    def test_wrong_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            pack_witness(np.zeros(10, np.uint8), 4)


class PackZLincheckTest(parameterized.TestCase):
    @parameterized.parameters((13, 6), (16, 10), (16, 8))
    def test_byte_layout_matches_consumer(self, m: int, k_log: int) -> None:
        # Rebuild the witness both packers derive from, then read the lincheck
        # bytes exactly as partial_fold_packed_z ingests them: reshape (n_bytes, k),
        # bit r of byte[byte_idx, i_inner] == z[i_inner + (8·byte_idx + r)·k].
        w = _rand_bits(m * 100 + k_log, 1 << m)
        zlc = pack_z_lincheck_from_packed(pack_witness(w, m), m, k_log)
        k = 1 << k_log
        n_bytes = (1 << m) // k // 8
        stripes = np.frombuffer(zlc, np.uint8).reshape(n_bytes, k)
        for byte_idx in range(min(n_bytes, 4)):
            for i_inner in range(min(k, 6)):
                byte = int(stripes[byte_idx, i_inner])
                for r in range(8):
                    want = int(w[i_inner + (8 * byte_idx + r) * k])
                    self.assertEqual((byte >> r) & 1, want)

    def test_length_and_divisibility(self) -> None:
        zp = pack_witness(_rand_bits(1, 1 << 13), 13)
        self.assertLen(pack_z_lincheck_from_packed(zp, 13, 6), (1 << 13) // 8)
        with self.assertRaises(ValueError):
            pack_z_lincheck_from_packed(zp, 13, 11)  # n_outer = 4, not byte-aligned


if __name__ == "__main__":
    absltest.main()
