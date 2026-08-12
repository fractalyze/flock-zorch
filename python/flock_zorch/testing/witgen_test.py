# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Native unit test for `flock_zorch.witgen` (no golden).

Independent checks, so a layout slip and a math slip cannot mask each other:

1. The R1CS relation itself: `z == a AND b` for every bit of every block.
   ADD rows store `carry = left AND right`, lin-id rows pin `v` against
   all-ones, the prefix is pinned the same way, and the padding is zero on
   all three streams — so the identity must hold wholesale, with no
   reference implementation in the loop.
2. A scalar big-int reference built field-by-field from the layout spec
   (append-only cursor, no word packing tricks) must reproduce the packed
   words exactly. This pins the bit layout without golden files.
3. `extract_inputs` round-trips the inputs back out of the packed z.
4. The lincheck stripe must byte-match `pcs.pack.pack_z_lincheck_from_packed`
   — the separately-tested HOST port of the same flock function — including
   the zero tail past the useful bits (the `cargo test` full-stripe
   contract; release flock leaves it unwritten and the fold never reads it).
   This is also the pin that keeps the device and host ports from drifting
   apart across a flock pin bump.
5. The compression math is anchored against `hash_frx.blake3.compress`, an
   implementation this module shares no round/emission code with: the
   out_lo/out_hi regions of `z` must equal its 16 output words (words 0..8
   are `state ^ state>>8`, words 8..16 are `state[8..] ^ cv`).
6. `blocks_from_seed` must match a scalar splitmix64 transcription of the
   challenge harness's `generate_compressions` (sequential state, 25 draws
   per compression) — the device version replaces the running state with a
   closed form, and this pins the two equal.

6. On GPU, the Pallas kernel must byte-match the portable XLA emission. The
   two share only the layout constants, so they are independent implementations
   of the same spec — which is what makes the comparison a gate rather than a
   tautology. Skipped off GPU: Triton has no CPU lowering.

Inputs are random u32s — block_len and flags included, so the flags high bit
(the streaming writer's pending-bit path in the Rust source) is exercised.
"""
from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest
from hash_frx.blake3.compress import IV, compress

from flock_zorch import witgen
from flock_zorch.pcs import pack


def _rand_inputs(rng, n):
    cv = rng.integers(0, 1 << 32, size=(n, 8), dtype=np.uint32)
    m = rng.integers(0, 1 << 32, size=(n, 16), dtype=np.uint32)
    counter = rng.integers(0, 1 << 63, size=(n,), dtype=np.uint64)
    block_len = rng.integers(0, 1 << 32, size=(n,), dtype=np.uint32)
    flags = rng.integers(0, 1 << 32, size=(n,), dtype=np.uint32)
    return cv, m, counter, block_len, flags


def _ref_streams(cv, m, counter, block_len, flags):
    """Scalar big-int reference for one block: (z, a, b) as K-bit ints."""
    mask31 = (1 << 31) - 1
    ones32 = (1 << 32) - 1

    def add_row(x, y):
        s = (x + y) & ones32
        cin = s ^ x ^ y
        left = (x ^ cin) & mask31
        right = (y ^ cin) & mask31
        return s, left, right, left & right

    def ror(x, r):
        return ((x >> r) | (x << (32 - r))) & ones32

    t_lo = int(counter) & ones32
    t_hi = int(counter) >> 32
    state = list(map(int, cv)) + list(IV[:4]) + [t_lo, t_hi, int(block_len), int(flags)]

    acc = {"z": 0, "a": 0, "b": 0}
    pos = {"z": 0, "a": 0, "b": 0}

    def put(stream, value, nbits):
        acc[stream] |= (value & ((1 << nbits) - 1)) << pos[stream]
        pos[stream] += nbits

    def put_lin(value, nbits=32):
        put("z", value, nbits)
        put("a", value, nbits)
        put("b", (1 << nbits) - 1, nbits)

    for w in range(8):
        put_lin(int(cv[w]))
    out_lo_pos = pos["z"]
    for _ in range(8):  # out_lo: reserved, back-filled after the rounds
        put_lin(0)
    put_lin(1, 1)
    for w in range(16):
        put_lin(int(m[w]))
    for v in (t_lo, t_hi, int(block_len), int(flags)):
        put_lin(v)

    for r in range(witgen.ROUNDS):
        sched = witgen._SCHEDULE[r]
        for g in range(8):
            la, lb, lc, ld = witgen._G_LANES[g]
            mx, my = int(m[sched[2 * g]]), int(m[sched[2 * g + 1]])
            a0, b0, c0, d0 = state[la], state[lb], state[lc], state[ld]
            rows = []
            tmp0, l, rr, cy = add_row(a0, b0)
            rows.append((cy, l, rr))
            a1, l, rr, cy = add_row(tmp0, mx)
            rows.append((cy, l, rr))
            d1 = ror(d0 ^ a1, 16)
            c1, l, rr, cy = add_row(c0, d1)
            rows.append((cy, l, rr))
            b1 = ror(b0 ^ c1, 12)
            tmp1, l, rr, cy = add_row(a1, b1)
            rows.append((cy, l, rr))
            a2, l, rr, cy = add_row(tmp1, my)
            rows.append((cy, l, rr))
            d2 = ror(d1 ^ a2, 8)
            c2, l, rr, cy = add_row(c1, d2)
            rows.append((cy, l, rr))
            b_new = ror(b1 ^ c2, 7)
            for cy, l, rr in rows:
                put("z", cy, 31)
                put("a", l, 31)
                put("b", rr, 31)
            put_lin(b_new)
            put_lin(d2)
            state[la], state[lb], state[lc], state[ld] = a2, b_new, c2, d2

    for w in range(8):
        put_lin(state[w + 8] ^ int(cv[w]))
    assert pos["z"] == witgen.USEFUL_BITS

    for w in range(8):  # back-fill the reserved out_lo slot
        v = state[w] ^ state[w + 8]
        p = out_lo_pos + 32 * w
        acc["z"] |= v << p
        acc["a"] |= v << p
        # b already carries all-ones there from the reservation pass.
    return acc["z"], acc["a"], acc["b"]


def _words(x):
    return np.array(
        [(x >> (64 * i)) & ((1 << 64) - 1) for i in range(witgen.WORDS_PER_BLOCK)],
        dtype=np.uint64,
    )


class WitgenTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        rng = np.random.default_rng(0xF17C)
        cls.inputs = _rand_inputs(rng, 8)
        z, a, b = witgen.witness_blake3(*cls.inputs)
        cls.z, cls.a, cls.b = map(np.asarray, (z, a, b))
        cls.refs = [_ref_streams(*(x[i] for x in cls.inputs)) for i in range(8)]

    def test_r1cs_relation_holds(self):
        np.testing.assert_array_equal(self.z, self.a & self.b)

    def test_matches_scalar_reference(self):
        for i, (rz, ra, rb) in enumerate(self.refs):
            np.testing.assert_array_equal(self.z[i], _words(rz), f"z block {i}")
            np.testing.assert_array_equal(self.a[i], _words(ra), f"a block {i}")
            np.testing.assert_array_equal(self.b[i], _words(rb), f"b block {i}")

    def test_pallas_kernel_matches_portable_emission(self):
        if frx.default_backend() != "gpu":
            self.skipTest("the Pallas kernel has no CPU lowering")
        from flock_zorch import witgen_pallas

        # Same batch size the fixture uses, deliberately: a Triton compile of
        # this kernel costs ~390 s per distinct shape, and reusing the shape
        # `setUpClass` already compiled keeps the check nearly free. 8 rows is
        # not a multiple of the 16-row tile, so it exercises the pad-and-slice
        # path — the exact-tile path is the same code with pad == 0.
        inputs = _rand_inputs(np.random.default_rng(0xA11A5), 8)
        want = witgen._witness_blake3_xla(*inputs)
        got = witgen_pallas.witness_blake3(*inputs)
        for name, w, g in zip("zab", want, got):
            np.testing.assert_array_equal(
                np.asarray(w), np.asarray(g), f"{name} stream"
            )

    def test_extract_inputs_roundtrip(self):
        got = witgen.extract_inputs(self.z.reshape(-1, 2))
        for name, want, have in zip(
            ("cv", "m", "counter", "block_len", "flags"), self.inputs, got
        ):
            np.testing.assert_array_equal(have, want, name)

    def test_lincheck_stripe_matches_host_port(self):
        # 8 blocks of 2^14 bits = one stripe group = a 2^17-bit witness.
        want = np.frombuffer(
            pack.pack_z_lincheck_from_packed(self.z.reshape(-1, 2), 17, witgen.K_LOG),
            np.uint8,
        ).reshape(1, witgen.STRIPE_BYTES_PER_GROUP)
        got = np.asarray(witgen.lincheck_stripe(self.z))
        np.testing.assert_array_equal(got, want)

    def test_blocks_from_seed_matches_splitmix_reference(self):
        log2, seed = 8, 0xDEAD_BEEF_1234_5678
        mask = (1 << 64) - 1
        s = seed ^ ((log2 << 29) & mask)
        draws = []
        for _ in range((1 << log2) * 25):
            s = (s + 0x9E37_79B9_7F4A_7C15) & mask
            z = s
            z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & mask
            draws.append((z ^ (z >> 31)) & 0xFFFF_FFFF)
        want = np.array(draws, np.uint32).reshape(-1, 25)
        cv, m, counter, block_len, flags = witgen.blocks_from_seed(
            np.uint64(seed), log2
        )
        np.testing.assert_array_equal(np.asarray(cv), want[:, 0:8])
        np.testing.assert_array_equal(np.asarray(m), want[:, 8:24])
        np.testing.assert_array_equal(
            np.asarray(counter), want[:, 24].astype(np.uint64)
        )
        np.testing.assert_array_equal(np.asarray(block_len), np.full(1 << log2, 64))
        np.testing.assert_array_equal(np.asarray(flags), np.full(1 << log2, 11))

    def test_out_regions_match_hash_frx_compress(self):
        cv, m, counter, block_len, flags = self.inputs
        ctr = np.stack(
            [counter.astype(np.uint32), (counter >> np.uint64(32)).astype(np.uint32)],
            axis=1,
        )
        out = np.asarray(compress(cv, m, ctr, block_len, flags))
        lo_start, hi_start = 8 * 32, witgen.USEFUL_BITS - 8 * 32
        z_bits = np.unpackbits(
            self.z.view(np.uint8).reshape(self.z.shape[0], -1),
            axis=1,
            bitorder="little",
        )
        for i in range(cv.shape[0]):
            lo = np.packbits(
                z_bits[i, lo_start : lo_start + 256], bitorder="little"
            ).view(np.uint32)
            hi = np.packbits(
                z_bits[i, hi_start : hi_start + 256], bitorder="little"
            ).view(np.uint32)
            np.testing.assert_array_equal(lo, out[i, :8], f"out_lo block {i}")
            np.testing.assert_array_equal(hi, out[i, 8:], f"out_hi block {i}")


if __name__ == "__main__":
    absltest.main()
