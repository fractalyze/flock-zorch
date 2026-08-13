# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Native unit test for `flock_zorch.r1cs_hashes.sha2_witness` (no golden).

Independent checks, so a layout slip and a math slip cannot mask each other:

1. The R1CS relation itself: `z == a AND b` for every bit of every block. ADD
   rows store `carry = left AND right`, lin-id rows pin `v` against all-ones,
   AND rows store `x & y` against `x` and `y`, the constant wire is `(1, 1, 1)`
   and the padding is zero on all three streams — so the identity must hold
   wholesale, with no reference implementation in the loop.
2. A scalar big-int reference built field-by-field from the layout spec (an
   offset-addressed scatter over Python ints, sharing no word-packing code with
   `_emit`) must reproduce the packed words exactly. sha2 packs tightly, so
   fields straddle 64-bit boundaries and this is what pins the shifts.
3. The compression math is anchored against `hash_frx.sha256.compress`, an
   implementation this module shares no round or schedule code with: the
   `H_out` region of `z` must equal its output on the same inputs. That
   transitively pins the round constants, both sigma functions, the message
   schedule, and the output feed-forward.
4. `extract_inputs` round-trips the inputs back out of the packed z.
5. The region bases are pinned to their literal values. They are derived from
   the field counts rather than hardcoded in the module, so a wrong count would
   otherwise slide every later region silently.

Inputs are random u32s.
"""
from __future__ import annotations

import functools

import numpy as np
from absl.testing import absltest
from hash_frx.sha256 import compress

from flock_zorch.r1cs_hashes import sha2_witness as ws

_N_BLOCKS = 4
_ONES32 = (1 << 32) - 1
_MASK31 = (1 << 31) - 1


def _ref_streams(h_in, m):
    """Scalar big-int reference for one block: (z, a, b) as K-bit ints.

    Deliberately unlike `_emit`: every field is scattered into one big integer
    at its own bit offset, so the two share no packing logic.
    """
    acc = {"z": 0, "a": 0, "b": 0}

    def put(off, width, zv, av, bv):
        for key, v in (("z", zv), ("a", av), ("b", bv)):
            acc[key] |= (v & ((1 << width) - 1)) << off

    def put_lin(off, v):
        put(off, 32, v, v, _ONES32)

    def put_and(off, x, y):
        put(off, 32, x & y, x, y)

    def add(off, x, y):
        s = (x + y) & _ONES32
        cin = s ^ x ^ y
        left = (x ^ cin) & _MASK31
        right = (y ^ cin) & _MASK31
        put(off, 31, left & right, left, right)
        return s

    def rotr(x, r):
        return ((x >> r) | (x << (32 - r))) & _ONES32

    h = [int(v) for v in h_in]
    w = [int(v) for v in m]
    for i in range(8):
        put_lin(ws.H_BASE + 32 * i, h[i])
    for i in range(16):
        put_lin(ws.M_BASE + 32 * i, w[i])

    for t in range(16, 64):
        j = t - 16
        x2, x7, x15, x16 = w[t - 2], w[t - 7], w[t - 15], w[t - 16]
        s1 = rotr(x2, 17) ^ rotr(x2, 19) ^ (x2 >> 10)
        s0 = rotr(x15, 7) ^ rotr(x15, 18) ^ (x15 >> 3)
        base = ws.SCHED_CARRY_BASE + j * ws.SCHED_CARRY_STRIDE
        val = s1
        for i, operand in enumerate((x7, s0, x16)):
            val = add(base + i * 31, val, operand)
        put_lin(ws.W_BASE + 32 * j, val)
        w.append(val)

    st = list(h)
    for r in range(64):
        aa, bb, cc, dd, ee, ff, gg, hh = st
        f_xor_g = ff ^ gg
        put_and(ws.CH_AND_BASE + 32 * r, ee, f_xor_g)
        ch_out = (ee & f_xor_g) ^ gg
        b_xor_a, c_xor_a = bb ^ aa, cc ^ aa
        put_and(ws.MAJ_AND_BASE + 32 * r, b_xor_a, c_xor_a)
        maj_out = (b_xor_a & c_xor_a) ^ aa

        s1e = rotr(ee, 6) ^ rotr(ee, 11) ^ rotr(ee, 25)
        s0a = rotr(aa, 2) ^ rotr(aa, 13) ^ rotr(aa, 22)
        base = ws.ROUND_CARRY_BASE + r * ws.ROUND_CARRY_STRIDE
        t1 = add(base, hh, s1e)
        t1 = add(base + 31, t1, ch_out)
        t1 = add(base + 62, t1, ws._K[r])
        t1 = add(base + 93, t1, w[r])
        t2 = add(base + 124, s0a, maj_out)
        e_new = add(base + 155, dd, t1)
        a_new = add(base + 186, t1, t2)

        put_lin(ws.T1_BASE + 32 * r, t1)
        put_lin(ws.E_NEW_BASE + 32 * r, e_new)
        put_lin(ws.A_NEW_BASE + 32 * r, a_new)
        st = [a_new, aa, bb, cc, e_new, ee, ff, gg]

    for i in range(8):
        out = add(ws.OUT_CARRY_BASE + 31 * i, st[i], h[i])
        put_lin(ws.H_OUT_BASE + 32 * i, out)

    put(ws.Z_CONST_POS, 1, 1, 1, 1)
    return acc["z"], acc["a"], acc["b"]


def _words(x):
    return np.array(
        [(x >> (64 * i)) & ((1 << 64) - 1) for i in range(ws.WORDS_PER_BLOCK)],
        dtype=np.uint64,
    )


@functools.lru_cache(maxsize=None)
def _emit():
    rng = np.random.default_rng(0x51A2)
    h_in = rng.integers(0, 1 << 32, size=(_N_BLOCKS, 8), dtype=np.uint32)
    m = rng.integers(0, 1 << 32, size=(_N_BLOCKS, 16), dtype=np.uint32)
    streams = tuple(np.asarray(x) for x in ws.witness_sha2(h_in, m))
    return (h_in, m) + streams


class WitgenSha2Test(absltest.TestCase):
    def test_r1cs_relation_holds(self):
        _, _, z, a, b = _emit()
        np.testing.assert_array_equal(z, a & b)

    def test_matches_scalar_reference(self):
        h_in, m, z, a, b = _emit()
        for i in range(_N_BLOCKS):
            rz, ra, rb = _ref_streams(h_in[i], m[i])
            np.testing.assert_array_equal(z[i], _words(rz), f"z block {i}")
            np.testing.assert_array_equal(a[i], _words(ra), f"a block {i}")
            np.testing.assert_array_equal(b[i], _words(rb), f"b block {i}")

    def test_h_out_region_matches_hash_frx_compress(self):
        h_in, m, z, _, _ = _emit()
        want = np.asarray(compress(h_in, m[:, None, :]))
        got = ws.read_words(z, ws.H_OUT_BASE, 8)
        np.testing.assert_array_equal(got, want)

    def test_extract_inputs_roundtrip(self):
        h_in, m, z, _, _ = _emit()
        got_h, got_m = ws.extract_inputs(z.reshape(-1, 2))
        np.testing.assert_array_equal(got_h, h_in)
        np.testing.assert_array_equal(got_m, m)

    def test_constant_wire_and_padding(self):
        _, _, *streams = _emit()
        w, s = divmod(ws.Z_CONST_POS, 64)
        for name, stream in zip("zab", streams):
            self.assertTrue(
                bool(np.all((stream[:, w] >> np.uint64(s)) & np.uint64(1))),
                f"{name} constant wire",
            )
        # Everything past the constant wire is pad, on all three streams.
        tail = np.uint64(~((1 << (s + 1)) - 1) & ((1 << 64) - 1))
        for name, stream in zip("zab", streams):
            np.testing.assert_array_equal(
                stream[:, w] & tail, np.zeros(_N_BLOCKS, np.uint64), f"{name} tail"
            )
            np.testing.assert_array_equal(
                stream[:, w + 1 :],
                np.zeros((_N_BLOCKS, ws.WORDS_PER_BLOCK - w - 1), np.uint64),
                f"{name} padding",
            )

    def test_region_bases_are_the_flock_layout(self):
        # Derived from the field counts in the module, so a wrong count would
        # slide every later region; these are the values flock's constants give.
        self.assertEqual(
            (
                ws.H_BASE,
                ws.H_OUT_BASE,
                ws.M_BASE,
                ws.CH_AND_BASE,
                ws.MAJ_AND_BASE,
                ws.ROUND_CARRY_BASE,
                ws.W_BASE,
                ws.SCHED_CARRY_BASE,
                ws.T1_BASE,
                ws.E_NEW_BASE,
                ws.A_NEW_BASE,
                ws.OUT_CARRY_BASE,
                ws.Z_CONST_POS,
                ws.USEFUL_BITS,
            ),
            (
                0,
                256,
                512,
                1024,
                3072,
                5120,
                19008,
                20544,
                25008,
                27056,
                29104,
                31152,
                31400,
                31401,
            ),
        )


if __name__ == "__main__":
    absltest.main()
