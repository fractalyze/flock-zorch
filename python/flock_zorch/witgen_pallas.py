# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""One-kernel BLAKE3 R1CS witness emission.

Byte-identical to `witgen.witness_blake3`, produced by a single Pallas program
per tile of compressions instead of an XLA fusion per slice of the output.

Why a kernel. `witgen._emit` stacks all 256 output words of a stream at once,
which makes the stream one fusion, and that fusion has to keep the sequential
336-row chain live while it assembles them: `ptxas` puts 11 of the module's 19
kernels at the 255-register cap with up to 716 B/thread spilled, so 8 warps/SM,
and the emit runs at 289 GB/s against a 1453 GB/s store roofline. XLA cannot
express "write word `w` and forget it" -- its emitters materialize
per-output-element expressions -- so the liveness is structural rather than a
tuning miss (fractalyze/xla#495).

A kernel does express it. One lane per compression walks the chain once holding
the 16 state words, and stores each output word as soon as the last field
reaching it is produced. Fields are appended in strictly increasing bit
position, so at most one word is still accumulating at any moment.

The three streams share the walk -- z, a and b differ only in which value each
row contributes -- so the chain runs once instead of three times.

`out_lo` is the single field out of bit order: it sits at bits [256, 512) but
is only known after the last round. That range is exactly words 4..7 with no
straddling neighbour, so the walk skips it and those words are stored at the
end.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
from frx import lax
from frx.experimental import pallas as pl
from frx.experimental.pallas import triton as plgpu
from hash_frx.blake3.compress import IV, ROUNDS

from flock_zorch.witgen import (
    _G_LANES,
    _SCHEDULE,
    CARRY_BITS,
    STRIPE_BYTES_PER_GROUP,
    STRIPE_USEFUL_WORDS,
    WORDS_PER_BLOCK,
)

# Rows per program. Measured on an RTX 5090 writing this [N, 256] u64 shape
# with the per-word store this kernel uses: 16 rows reaches 1254 GB/s against a
# 1428 GB/s XLA reference, 64 rows falls to 643. Any divisor of the batch is
# correct; only the speed changes.
_BLOCK_ROWS = 16
_NUM_WARPS = 8

_MASK31 = 0x7FFF_FFFF
_ONES32 = 0xFFFF_FFFF
_MASK64 = (1 << 64) - 1

# `out_lo` covers bits [256, 512) = words 4..7 exactly, so skipping it costs no
# straddle handling.
_OUT_LO_BITS = 256

# Stripe groups per program. The transpose is cross-lane over a group of 8
# blocks, so a program holds whole groups.
_STRIPE_GROUPS = 8
_STRIPE_ROWS = _STRIPE_GROUPS * 8
# The kernel writes the group's full 16,384 bytes as u64, so the zero tail is
# stored rather than concatenated on afterwards -- padding outside would cost a
# second pass over the whole output.
_STRIPE_U64_PER_GROUP = STRIPE_BYTES_PER_GROUP // 8  # 2048
_STRIPE_DATA_U64 = STRIPE_USEFUL_WORDS * 8  # 1928, the rest is the zero pad


class _Writer:
    """Accumulates fields in bit order, storing each word once it is full.

    One accumulator per stream, not 256: because fields arrive in increasing
    bit position, a word can only receive bits until the walk passes its end.
    That is the whole point of the kernel -- it is what keeps the live set at
    the 16 state words plus three accumulators.
    """

    def __init__(self, refs):
        self.refs = refs
        self.pos = 0
        self.word = 0
        self.acc = [self._zero() for _ in refs]

    @staticmethod
    def _zero():
        return fnp.zeros((_BLOCK_ROWS,), fnp.uint64)

    def _store(self):
        for ref, acc in zip(self.refs, self.acc):
            ref[:, self.word] = acc

    def put(self, values, width: int):
        """One field: `values` is (z, a, b), each a uint32 array or an int."""
        shift = self.pos % 64
        room = 64 - shift
        straddles = width > room
        high = [None] * len(values)
        for i, v in enumerate(values):
            if isinstance(v, int):
                v &= (1 << width) - 1
                self.acc[i] = self.acc[i] | _u64(v << shift)
                if straddles:
                    # Safe as a plain literal: a field is at most 32 bits, so
                    # its high part is under 2^31 and never trips `_u64`.
                    high[i] = fnp.full((_BLOCK_ROWS,), v >> room, fnp.uint64)
            else:
                v64 = v.astype(fnp.uint64)
                self.acc[i] = self.acc[i] | (v64 << fnp.uint64(shift))
                if straddles:
                    high[i] = v64 >> fnp.uint64(room)
        self.pos += width
        if straddles or self.pos % 64 == 0:
            self._store()
            self.word += 1
            self.acc = [h if h is not None else self._zero() for h in high]

    def skip(self, bits: int):
        """Advance past a word-aligned range stored later (`out_lo`)."""
        assert self.pos % 64 == 0 and bits % 64 == 0, (self.pos, bits)
        self.pos += bits
        self.word += bits // 64

    def finish(self):
        """Store the partial word and zero-fill the padding to word 255."""
        if self.pos % 64:
            self._store()
            self.word += 1
        zero = self._zero()
        for w in range(self.word, WORDS_PER_BLOCK):
            for ref in self.refs:
                ref[:, w] = zero


def _u64(v: int):
    """A uint64 scalar constant.

    Triton builds a scalar through an MLIR integer attribute, which is signed,
    so anything at or above 2^63 raises `std::bad_cast` on the way in. The
    b-stream's all-ones masks land there whenever they sit in a word's high
    half. Assembling from two 32-bit halves keeps every literal in range.
    """
    v &= _MASK64
    if v < (1 << 63):
        return fnp.uint64(v)
    return fnp.uint64(v & _ONES32) | (fnp.uint64(v >> 32) << fnp.uint64(32))


def _rotr(x, k: int):
    return (x >> fnp.uint32(k)) | (x << fnp.uint32(32 - k))


def _kernel(cv_ref, m_ref, ctr_ref, bl_ref, fl_ref, z_ref, a_ref, b_ref):
    cv = [cv_ref[:, i] for i in range(8)]
    m = [m_ref[:, i] for i in range(16)]
    ctr = ctr_ref[:]
    t_lo = ctr.astype(fnp.uint32)
    t_hi = (ctr >> fnp.uint64(32)).astype(fnp.uint32)
    block_len, flags = bl_ref[:], fl_ref[:]

    state = list(cv)
    state += [fnp.full((_BLOCK_ROWS,), int(c), fnp.uint32) for c in IV[:4]]
    state += [t_lo, t_hi, block_len, flags]

    w = _Writer((z_ref, a_ref, b_ref))

    for v in cv:  # words 0..3
        w.put((v, v, _ONES32), 32)
    w.skip(_OUT_LO_BITS)  # words 4..7, stored after the chain
    w.put((1, 1, 1), 1)  # the constant-1 wire at bit 512
    for v in m:
        w.put((v, v, _ONES32), 32)
    for v in (t_lo, t_hi, block_len, flags):
        w.put((v, v, _ONES32), 32)

    def add(x, y):
        s = x + y
        cin = s ^ x ^ y
        left = (x ^ cin) & fnp.uint32(_MASK31)
        right = (y ^ cin) & fnp.uint32(_MASK31)
        w.put((left & right, left, right), CARRY_BITS)
        return s

    for r in range(ROUNDS):
        sched = _SCHEDULE[r]
        for g in range(8):
            la, lb, lc, ld = _G_LANES[g]
            mx, my = m[sched[2 * g]], m[sched[2 * g + 1]]
            a0, b0, c0, d0 = state[la], state[lb], state[lc], state[ld]

            a1 = add(add(a0, b0), mx)
            d1 = _rotr(d0 ^ a1, 16)
            c1 = add(c0, d1)
            b1 = _rotr(b0 ^ c1, 12)
            a2 = add(add(a1, b1), my)
            d2 = _rotr(d1 ^ a2, 8)
            c2 = add(c1, d2)
            b_new = _rotr(b1 ^ c2, 7)
            for v in (b_new, d2):
                w.put((v, v, _ONES32), 32)

            state[la], state[lb], state[lc], state[ld] = a2, b_new, c2, d2

    for i in range(8):  # out_hi
        v = state[i + 8] ^ cv[i]
        w.put((v, v, _ONES32), 32)
    w.finish()

    # out_lo, deferred: words 4..7, two u32 halves each. b carries the all-ones
    # mask here exactly as it does for every other value field.
    out_lo = [state[i] ^ state[i + 8] for i in range(8)]
    ones = fnp.full((_BLOCK_ROWS,), _ONES32, fnp.uint32)
    for j in range(4):
        lo, hi = out_lo[2 * j], out_lo[2 * j + 1]
        packed = lo.astype(fnp.uint64) | (hi.astype(fnp.uint64) << fnp.uint64(32))
        z_ref[:, 4 + j] = packed
        a_ref[:, 4 + j] = packed
        b_ref[:, 4 + j] = ones.astype(fnp.uint64) | (
            ones.astype(fnp.uint64) << fnp.uint64(32)
        )


@frx.jit
def witness_blake3(cv, m, counter, block_len, flags):
    """Packed z/a/b witness streams, byte-identical to `witgen.witness_blake3`.

    cv: uint32 [N, 8], m: uint32 [N, 16], counter: uint64 [N],
    block_len/flags: uint32 [N] -> three uint64 [N, 256] arrays.

    GPU only — Triton has no CPU lowering. Reach this through
    `witgen.witness_blake3`, which keeps the portable path for everything else.
    """
    n = cv.shape[0]
    pad = -n % _BLOCK_ROWS  # the grid is whole programs, so fill a short tile
    if pad:
        cv, m, counter, block_len, flags = (
            fnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1))
            for x in (cv, m, counter, block_len, flags)
        )
    rows = n + pad
    out = frx.ShapeDtypeStruct((rows, WORDS_PER_BLOCK), fnp.uint64)
    row = lambda i: (i,)  # noqa: E731
    z, a, b = pl.pallas_call(
        _kernel,
        out_shape=(out, out, out),
        grid=(rows // _BLOCK_ROWS,),
        in_specs=[
            pl.BlockSpec((_BLOCK_ROWS, 8), lambda i: (i, 0)),
            pl.BlockSpec((_BLOCK_ROWS, 16), lambda i: (i, 0)),
            pl.BlockSpec((_BLOCK_ROWS,), row),
            pl.BlockSpec((_BLOCK_ROWS,), row),
            pl.BlockSpec((_BLOCK_ROWS,), row),
        ],
        out_specs=[pl.BlockSpec((_BLOCK_ROWS, WORDS_PER_BLOCK), lambda i: (i, 0))] * 3,
        compiler_params=plgpu.CompilerParams(num_warps=_NUM_WARPS),
        name="blake3_r1cs_witness",
    )(cv, m, counter, block_len, flags)
    return (z[:n], a[:n], b[:n]) if pad else (z, a, b)


def _transpose_8x8_bits(x):
    """Hacker's Delight 7-3 on the 8x8 bit matrix in each u64 (byte i = row i).

    The device twin of `witgen._transpose_8x8_bits`; every mask is under 2^63,
    so none of them need `_u64`.
    """
    t = (x ^ (x >> fnp.uint64(7))) & fnp.uint64(0x00AA_00AA_00AA_00AA)
    x = x ^ t ^ (t << fnp.uint64(7))
    t = (x ^ (x >> fnp.uint64(14))) & fnp.uint64(0x0000_CCCC_0000_CCCC)
    x = x ^ t ^ (t << fnp.uint64(14))
    t = (x ^ (x >> fnp.uint64(28))) & fnp.uint64(0x0000_0000_F0F0_F0F0)
    return x ^ t ^ (t << fnp.uint64(28))


def _stripe_kernel(z_ref, out_ref):
    """One program's groups of the lincheck stripe.

    Per useful word, all eight byte columns at once: `[group, row, column]`.
    For column `c` the eight rows' bytes occupy disjoint byte positions of the
    packed u64, so the cross-row gather is a SUM rather than an OR -- Triton
    lowers the reduction, and disjointness makes them identical. Emitting one
    8-wide store per word rather than eight scalar ones matters: the scalar
    form did not finish compiling in half an hour.
    """
    shift = fnp.arange(8, dtype=fnp.uint64) * fnp.uint64(8)
    lane_shift = shift.reshape(1, 8, 1)
    for word in range(STRIPE_USEFUL_WORDS):
        rows = z_ref[:, word].reshape(_STRIPE_GROUPS, 8, 1)
        byte = (rows >> shift) & fnp.uint64(0xFF)
        packed = fnp.sum(byte << lane_shift, axis=1)
        out_ref[:, word * 8 : (word + 1) * 8] = _transpose_8x8_bits(packed)
    # The zero tail goes out in the same 8-wide stores: Triton requires every
    # array shape to be a power of two, and the tail is 120 u64 wide.
    zero = fnp.zeros((_STRIPE_GROUPS, 8), fnp.uint64)
    for word in range(STRIPE_USEFUL_WORDS, WORDS_PER_BLOCK):
        out_ref[:, word * 8 : (word + 1) * 8] = zero


@frx.jit
def lincheck_stripe(z):
    """The lincheck byte stripe from a packed z stream, byte-identical to
    `witgen.lincheck_stripe`.

    z: uint64 [N, 256] with N a multiple of `_STRIPE_ROWS` -> uint8
    [N/8, 16384]. GPU only; reach it through `witgen.lincheck_stripe`.

    Left as its own kernel rather than folded into the witness emission: folding
    was measured, and it cost the emit 27% of its throughput (765 -> 561 GB/s)
    because the cross-lane reduce competes for the chain's registers, so the
    deleted read bought only 0.44 ms while compile time went 390 -> 966 s.
    """
    n = z.shape[0]
    if n % _STRIPE_ROWS:
        raise ValueError(f"stripe kernel needs a multiple of {_STRIPE_ROWS} blocks")
    packed = pl.pallas_call(
        _stripe_kernel,
        out_shape=frx.ShapeDtypeStruct((n // 8, _STRIPE_U64_PER_GROUP), fnp.uint64),
        grid=(n // _STRIPE_ROWS,),
        in_specs=[pl.BlockSpec((_STRIPE_ROWS, WORDS_PER_BLOCK), lambda i: (i, 0))],
        out_specs=pl.BlockSpec(
            (_STRIPE_GROUPS, _STRIPE_U64_PER_GROUP), lambda i: (i, 0)
        ),
        compiler_params=plgpu.CompilerParams(num_warps=_NUM_WARPS),
        name="blake3_lincheck_stripe",
    )(z)
    # u64 entry (i*8 + c) bitcasts to the eight bytes at i*64 + c*8 + t, which
    # is flock's layout; a bitcast + reshape is free where a pad would not be.
    return lax.bitcast_convert_type(packed, fnp.uint8).reshape(
        n // 8, STRIPE_BYTES_PER_GROUP
    )


def stripe_rows() -> int:
    """Batch granularity `lincheck_stripe` requires, for the dispatcher."""
    return _STRIPE_ROWS
