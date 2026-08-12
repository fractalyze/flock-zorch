# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Native unit test for `flock_zorch.witgen_keccak` (no golden).

Runs against both members of the family — single keccak and keccak3 — since
they share an emitter and differ only in the lane map. Independent checks, so a
layout slip and a math slip cannot mask each other:

1. The R1CS relation itself: `z == a AND b` for every lane of every block.
   AND rows store `(~b1) & b2` against `~b1` and `b2`, lin-id rows pin `v`
   against all-ones, the constant wire is `(1, 1, 1)`, and both the slot gaps
   and the tail padding are zero on all three streams — so the identity holds
   wholesale, with no reference implementation in the loop.
2. The permutation math is anchored against `hash_frx.keccak.KeccakF1600`, an
   implementation this module shares no round code with (it carries the state
   as uint32 half-lanes on a 5x5 grid, not whole u64 lanes): every
   sub-permutation's `state_24` region must equal its output on that sub's
   `state_0`. This is also what pins keccak3's sub-permutations as independent
   rather than chained.
3. The chi wiring is pinned structurally, without recomputing it. Row `(x, y)`
   of round `r` names `bs[(x+1)%5 + 5y]` in `a` and `bs[(x+2)%5 + 5y]` in `b`,
   so `b` at row `(x, y)` must equal `~a` at row `((x+1)%5, y)` — the same
   `bs` lane reached two ways. That pins the within-round lane order and the
   round stride against a reimplementation of neither.
4. The layout constants must agree with the matching lincheck circuit's, which
   are the same flock layout reached independently (and pinned by the proof
   gates). The two modules derive them separately, so this catches drift.

The exact lane placement against flock itself is pinned by the proof-level
gates, which is where a byte-identity claim belongs; these checks are what run
without a golden present.
"""
from __future__ import annotations

import functools

import frx
import numpy as np
from absl.testing import absltest, parameterized
from hash_frx.keccak.permutation import KeccakF1600

from flock_zorch import witgen_keccak as wk
from flock_zorch.lincheck import keccak as lincheck_keccak
from flock_zorch.lincheck import keccak3 as lincheck_keccak3

_N_BLOCKS = 4
_ONES = np.uint64(0xFFFF_FFFF_FFFF_FFFF)


def _to_halves(state):
    """[..., 25] u64 -> [..., 50] u32, lane `i` at elements `2i` (lo), `2i+1`."""
    lo = state.astype(np.uint32)
    hi = (state >> np.uint64(32)).astype(np.uint32)
    return np.stack([lo, hi], axis=-1).reshape(*state.shape[:-1], 2 * state.shape[-1])


def _from_halves(halves):
    """The inverse of `_to_halves`."""
    h = np.asarray(halves).reshape(*halves.shape[:-1], -1, 2)
    return h[..., 0].astype(np.uint64) | (h[..., 1].astype(np.uint64) << np.uint64(32))


@functools.lru_cache(maxsize=None)
def _emit(n_sub: int):
    """(state0 [N, n_sub, 25], z, a, b) for the family member of this width."""
    spec = wk.KECCAK if n_sub == 1 else wk.KECCAK3
    rng = np.random.default_rng(0xC0FFEE + n_sub)
    state0 = rng.integers(
        0, 1 << 64, size=(_N_BLOCKS, n_sub, wk.N_LANES), dtype=np.uint64
    )
    fn = wk.witness_keccak if n_sub == 1 else wk.witness_keccak3
    arg = state0[:, 0, :] if n_sub == 1 else state0
    return (spec, state0) + tuple(np.asarray(x) for x in fn(arg))


_CIRCUITS = (
    {"testcase_name": "_keccak", "n_sub": 1},
    {"testcase_name": "_keccak3", "n_sub": 3},
)


class WitgenKeccakTest(parameterized.TestCase):
    @parameterized.named_parameters(*_CIRCUITS)
    def test_r1cs_relation_holds(self, n_sub):
        _, _, z, a, b = _emit(n_sub)
        np.testing.assert_array_equal(z, a & b)

    @parameterized.named_parameters(*_CIRCUITS)
    def test_shape_is_one_block_per_group(self, n_sub):
        spec, _, *streams = _emit(n_sub)
        for name, s in zip("zab", streams):
            self.assertEqual(s.shape, (_N_BLOCKS, spec.words_per_block), name)

    @parameterized.named_parameters(*_CIRCUITS)
    def test_state0_regions_are_the_input(self, n_sub):
        spec, state0, z, a, b = _emit(n_sub)
        for i in range(n_sub):
            lo = spec.state0_lane(i)
            lanes = slice(lo, lo + wk.N_LANES)
            np.testing.assert_array_equal(z[:, lanes], state0[:, i, :], f"sub {i}")
            np.testing.assert_array_equal(a[:, lanes], state0[:, i, :], f"sub {i}")
            np.testing.assert_array_equal(
                b[:, lanes], np.full_like(state0[:, i, :], _ONES), f"sub {i}"
            )

    @parameterized.named_parameters(*_CIRCUITS)
    def test_state24_regions_match_hash_frx_permutation(self, n_sub):
        spec, state0, z, a, _ = _emit(n_sub)
        flat = state0.reshape(-1, wk.N_LANES)
        want = _from_halves(
            np.asarray(
                frx.vmap(KeccakF1600().permute)(frx.device_put(_to_halves(flat)))
            )
        ).reshape(_N_BLOCKS, n_sub, wk.N_LANES)
        for i in range(n_sub):
            lo = spec.state24_lane(i)
            lanes = slice(lo, lo + wk.N_LANES)
            np.testing.assert_array_equal(z[:, lanes], want[:, i, :], f"sub {i}")
            np.testing.assert_array_equal(a[:, lanes], want[:, i, :], f"sub {i}")

    @parameterized.named_parameters(*_CIRCUITS)
    def test_constant_wire(self, n_sub):
        spec, _, *streams = _emit(n_sub)
        for name, s in zip("zab", streams):
            np.testing.assert_array_equal(
                s[:, spec.z_const_lane], np.ones(_N_BLOCKS, np.uint64), name
            )

    @parameterized.named_parameters(*_CIRCUITS)
    def test_slot_gaps_and_padding_are_zero(self, n_sub):
        spec, _, *streams = _emit(n_sub)
        # Each 2,048-bit slot holds a 1,600-bit state, so seven lanes of every
        # slot are pad; everything past the last t row is pad as well.
        gaps = [
            slice(spec.state0_lane(i) + wk.N_LANES, spec.state24_lane(i))
            for i in range(n_sub)
        ]
        gaps += [
            slice(spec.state24_lane(i) + wk.N_LANES, spec.state0_lane(i + 1))
            for i in range(n_sub - 1)
        ]
        gaps.append(slice(spec.state24_lane(n_sub - 1) + wk.N_LANES, spec.z_const_lane))
        gaps.append(slice(spec.useful_lanes, spec.words_per_block))
        for name, s in zip("zab", streams):
            for g in gaps:
                np.testing.assert_array_equal(
                    s[:, g],
                    np.zeros((_N_BLOCKS, g.stop - g.start), np.uint64),
                    f"{name} stream, lanes [{g.start}, {g.stop})",
                )

    @parameterized.named_parameters(*_CIRCUITS)
    def test_chi_operands_name_the_same_lane_twice(self, n_sub):
        spec, _, _, a, b = _emit(n_sub)
        # b at row (x, y) and a at row ((x+1)%5, y) are both bs[(x+2)%5 + 5y].
        for i in range(n_sub):
            for r in range(wk.N_ROUNDS):
                base = spec.t_lane(i, r)
                for y in range(5):
                    for x in range(5):
                        np.testing.assert_array_equal(
                            b[:, base + wk._li(x, y)],
                            ~a[:, base + wk._li((x + 1) % 5, y)],
                            f"sub {i}, round {r}, lane ({x}, {y})",
                        )

    @parameterized.named_parameters(*_CIRCUITS)
    def test_layout_agrees_with_the_lincheck_circuit(self, n_sub):
        spec = wk.KECCAK if n_sub == 1 else wk.KECCAK3
        lk = lincheck_keccak if n_sub == 1 else lincheck_keccak3
        self.assertEqual(spec.k_log, lk.K_LOG)
        self.assertEqual(spec.words_per_block * wk.LANE_BITS, lk.K)
        self.assertEqual(wk.SLOT_BITS, lk.SLOT_BITS)
        self.assertEqual(spec.z_const_lane * wk.LANE_BITS, lk.Z_CONST)
        self.assertEqual(spec.t_lane_base * wk.LANE_BITS, lk.T_PACKED_BIT_BASE)
        if n_sub > 1:
            self.assertEqual(spec.n_sub, lk.N_SUB)

    def test_single_keccak_layout_agrees_with_its_lincheck_bases(self):
        # Only the single-keccak circuit names the state slots directly; the
        # keccak3 walker derives its own from N_SUB.
        lk = lincheck_keccak
        self.assertEqual(wk.N_LANES, lk.N_LANES)
        self.assertEqual(wk.LANE_BITS, lk.LANE_BITS)
        self.assertEqual(wk.N_ROUNDS, lk.N_T)
        self.assertEqual(wk.KECCAK.state0_lane(0) * wk.LANE_BITS, lk.STATE0_BIT_BASE)
        self.assertEqual(wk.KECCAK.state24_lane(0) * wk.LANE_BITS, lk.STATE24_BIT_BASE)
        self.assertEqual(list(wk._RC), lk.ROUND_CONSTANTS)
        # rho's offsets, reached from hash_frx here and from flock's Rust there.
        for y in range(5):
            for x in range(5):
                self.assertEqual(
                    wk.ROTATION_OFFSETS[wk._li(x, y)],
                    lk.RHO_OFFSETS[x][y],
                    f"rho offset ({x}, {y})",
                )


if __name__ == "__main__":
    absltest.main()
