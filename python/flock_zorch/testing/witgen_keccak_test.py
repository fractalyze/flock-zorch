# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Native unit test for `flock_zorch.witgen_keccak` (no golden).

Independent checks, so a layout slip and a math slip cannot mask each other:

1. The R1CS relation itself: `z == a AND b` for every lane of every block.
   AND rows store `(~b1) & b2` against `~b1` and `b2`, lin-id rows pin `v`
   against all-ones, the constant wire is `(1, 1, 1)`, and both the slot gaps
   and the tail padding are zero on all three streams — so the identity holds
   wholesale, with no reference implementation in the loop.
2. The permutation math is anchored against `hash_frx.keccak.KeccakF1600`, an
   implementation this module shares no round code with (it carries the state
   as uint32 half-lanes on a 5x5 grid, not whole u64 lanes): the `state_24`
   region of `z` must equal its output on the same `state_0`.
3. The chi wiring is pinned structurally, without recomputing it. Row `(x, y)`
   of round `r` names `bs[(x+1)%5 + 5y]` in `a` and `bs[(x+2)%5 + 5y]` in `b`,
   so `b` at row `(x, y)` must equal `~a` at row `((x+1)%5, y)` — the same
   `bs` lane reached two ways. That pins the within-round lane order and the
   25-lane round stride against a reimplementation of neither.
4. The layout constants must agree with `lincheck.keccak`'s, which are the
   same flock layout reached independently (and pinned by the keccak proof
   gate). The two modules derive them separately, so this catches drift.

The exact lane placement against flock itself is pinned by the proof-level
keccak gate, which is where a byte-identity claim belongs; these checks are
what run without a golden present.
"""
from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest
from hash_frx.keccak.permutation import KeccakF1600

from flock_zorch import witgen_keccak as wk
from flock_zorch.lincheck import keccak as lincheck_keccak

_N_BLOCKS = 4
_ONES = np.uint64(0xFFFF_FFFF_FFFF_FFFF)


def _to_halves(state):
    """[N, 25] u64 -> [N, 50] u32, lane `i` at elements `2i` (lo), `2i+1` (hi)."""
    lo = state.astype(np.uint32)
    hi = (state >> np.uint64(32)).astype(np.uint32)
    return np.stack([lo, hi], axis=2).reshape(state.shape[0], 2 * state.shape[1])


def _from_halves(halves):
    """The inverse of `_to_halves`."""
    h = np.asarray(halves).reshape(halves.shape[0], -1, 2)
    return h[:, :, 0].astype(np.uint64) | (
        h[:, :, 1].astype(np.uint64) << np.uint64(32)
    )


class WitgenKeccakTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        rng = np.random.default_rng(0xC0FFEE)
        cls.state0 = rng.integers(
            0, 1 << 64, size=(_N_BLOCKS, wk.N_LANES), dtype=np.uint64
        )
        z, a, b = wk.witness_keccak(cls.state0)
        cls.z, cls.a, cls.b = (np.asarray(x) for x in (z, a, b))

    def test_r1cs_relation_holds(self):
        np.testing.assert_array_equal(self.z, self.a & self.b)

    def test_shape_is_one_block_per_permutation(self):
        for name, s in zip("zab", (self.z, self.a, self.b)):
            self.assertEqual(s.shape, (_N_BLOCKS, wk.WORDS_PER_BLOCK), name)

    def test_state0_region_is_the_input(self):
        lanes = slice(wk.STATE0_LANE, wk.STATE0_LANE + wk.N_LANES)
        np.testing.assert_array_equal(self.z[:, lanes], self.state0)
        np.testing.assert_array_equal(self.a[:, lanes], self.state0)
        np.testing.assert_array_equal(
            self.b[:, lanes], np.full_like(self.state0, _ONES)
        )

    def test_state24_region_matches_hash_frx_permutation(self):
        want = _from_halves(
            np.asarray(
                frx.vmap(KeccakF1600().permute)(frx.device_put(_to_halves(self.state0)))
            )
        )
        lanes = slice(wk.STATE24_LANE, wk.STATE24_LANE + wk.N_LANES)
        np.testing.assert_array_equal(self.z[:, lanes], want)
        np.testing.assert_array_equal(self.a[:, lanes], want)

    def test_constant_wire(self):
        for name, s in zip("zab", (self.z, self.a, self.b)):
            np.testing.assert_array_equal(
                s[:, wk.Z_CONST_LANE], np.ones(_N_BLOCKS, np.uint64), name
            )

    def test_slot_gaps_and_padding_are_zero(self):
        # The two 2,048-bit slots hold a 1,600-bit state each, so seven lanes of
        # every slot are pad; everything past the last t row is pad as well.
        gaps = (
            slice(wk.STATE0_LANE + wk.N_LANES, wk.STATE24_LANE),
            slice(wk.STATE24_LANE + wk.N_LANES, wk.Z_CONST_LANE),
            slice(wk.USEFUL_LANES, wk.WORDS_PER_BLOCK),
        )
        for name, s in zip("zab", (self.z, self.a, self.b)):
            for g in gaps:
                np.testing.assert_array_equal(
                    s[:, g],
                    np.zeros((_N_BLOCKS, g.stop - g.start), np.uint64),
                    f"{name} stream, lanes [{g.start}, {g.stop})",
                )

    def test_chi_operands_name_the_same_lane_twice(self):
        # b at row (x, y) and a at row ((x+1)%5, y) are both bs[(x+2)%5 + 5y].
        for r in range(wk.N_ROUNDS):
            base = wk.T_LANE_BASE + r * wk.N_LANES
            for y in range(5):
                for x in range(5):
                    np.testing.assert_array_equal(
                        self.b[:, base + wk._li(x, y)],
                        ~self.a[:, base + wk._li((x + 1) % 5, y)],
                        f"round {r}, lane ({x}, {y})",
                    )

    def test_layout_agrees_with_the_lincheck_circuit(self):
        lk = lincheck_keccak
        self.assertEqual(wk.K_LOG, lk.K_LOG)
        self.assertEqual(wk.K, lk.K)
        self.assertEqual(wk.N_LANES, lk.N_LANES)
        self.assertEqual(wk.LANE_BITS, lk.LANE_BITS)
        self.assertEqual(wk.N_ROUNDS, lk.N_T)
        self.assertEqual(wk.SLOT_BITS, lk.SLOT_BITS)
        self.assertEqual(wk.STATE0_LANE * wk.LANE_BITS, lk.STATE0_BIT_BASE)
        self.assertEqual(wk.STATE24_LANE * wk.LANE_BITS, lk.STATE24_BIT_BASE)
        self.assertEqual(wk.Z_CONST_LANE * wk.LANE_BITS, lk.Z_CONST)
        self.assertEqual(wk.T_LANE_BASE * wk.LANE_BITS, lk.T_PACKED_BIT_BASE)
        # rho's offsets and iota's constants, reached from hash_frx here and
        # transcribed from flock's Rust there.
        self.assertEqual(list(wk._RC), lk.ROUND_CONSTANTS)
        for y in range(5):
            for x in range(5):
                self.assertEqual(
                    wk.ROTATION_OFFSETS[wk._li(x, y)],
                    lk.RHO_OFFSETS[x][y],
                    f"rho offset ({x}, {y})",
                )


if __name__ == "__main__":
    absltest.main()
