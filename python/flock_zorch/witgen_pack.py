# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Bit-packing shared by the tightly-packed R1CS witness circuits.

`witgen` (BLAKE3) and `witgen_sha2` both pack their fields tightly, so values
straddle 64-bit word boundaries and every field needs shift arithmetic. They
differ only in their layouts, which is what this module factors out: the
offset-addressed emitter and flock's ADD row.

The keccak family does not go through here. `witgen_keccak` places each state in
a 2,048-bit aligned slot and writes whole u64 lanes, so its "field list" is a
lane map with no shifts at all and its rows have no carries.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np

# The row vocabulary both circuits share. `a AND b = z` per bit, in three forms:
#
#     ADD carry (31 b)   z = left & right   a = left        b = right
#     lin-id    (32 b)   z = v              a = v           b = ONES32
#     AND       (32 b)   z = x & y          a = x           b = y
#
# sha2 is the only one of the four circuits that carries all three.
WORD_BITS = 32
CARRY_BITS = 31  # bit 31 is the discarded mod-2^32 carry-out and gets no slot
ONES32 = 0xFFFF_FFFF

_ONES64 = (1 << 64) - 1
_MASK31 = (1 << CARRY_BITS) - 1


def add_row(x, y):
    """One 32-bit ADD with its R1CS row parts: (sum, left, right, carry).

    flock stores an ADD row's addends with the carry-in folded in, writing them
    as `x ^ cin` and `y ^ cin` for `cin = sum ^ x ^ y`. The XOR self-cancels to
    `sum ^ y` and `sum ^ x`, so `cin` never needs computing.

    Worth folding by hand rather than transcribing flock and leaving it to the
    compiler: XLA's simplifier does not reassociate the chain to find the
    cancellation — optimized HLO keeps 10 xor instructions for the literal form
    against 4 for this one — and this runs 336 times per BLAKE3 block and 600
    per SHA-256 block, so it is runtime work on every element. The literal form
    survives in the tests, where it is the independent anchor that pins the
    identity.

    A plain int mask rather than `fnp.uint32(...)`: this body is traced both by
    XLA and, through `witgen_pallas`, by Triton, and a per-call `fnp` scalar
    would be an eager device constant on every one of those calls.
    """
    s = x + y
    left, right = (s ^ y) & _MASK31, (s ^ x) & _MASK31
    return s, left, right, left & right


def emit(fields, n: int, *, words: int, useful_bits: int):
    """Assemble one packed stream: [(offset, width, value)] -> uint64 [n, words].

    `value` is a uint32 [n] array (pre-masked to `width` bits, as the row
    builders guarantee) or a Python int constant. Array fields contribute to one
    output word, or two when they straddle a boundary, with shifts fixed at
    trace time; int fields accumulate into a single literal that seeds the
    constant words. Words past `useful_bits` are the block's zero padding and
    need no field.

    Offsets are explicit rather than a running cursor because a circuit does not
    have to produce its fields in bit order — sha2's `ch_and` lives at bit 1,024
    but is produced interleaved with that round's carries at 5,120 and up, and
    BLAKE3's `out_lo` sits at bit 256 but is only known after the last round. A
    cursor is the special case where the sorted offsets come out consecutive,
    which is what the coverage check below requires of every caller anyway.
    """
    pos = 0
    for off, width, _ in sorted(fields, key=lambda f: f[0]):
        # Explicit offsets mean a gap or an overlap in the field list would emit
        # a silently wrong stream; a cursor gets this for free.
        assert off == pos, f"field list leaves bit {pos} unwritten (next at {off})"
        pos += width
    assert pos == useful_bits, pos

    const_acc = 0
    contrib: list[list[tuple]] = [[] for _ in range(words)]
    for off, width, value in fields:
        if isinstance(value, int):
            const_acc |= (value & ((1 << width) - 1)) << off
            continue
        w, s = divmod(off, 64)
        contrib[w].append((value, s, False))
        if s + width > 64:
            contrib[w + 1].append((value, 64 - s, True))

    # A straddling field converts the same array in two adjacent words; widen
    # each distinct value once instead.
    widened: dict[int, object] = {}

    def as_u64(value):
        # Not `setdefault` — it evaluates its default eagerly, so the convert
        # would still be traced on every hit and the cache would buy nothing.
        key = id(value)
        if key not in widened:
            widened[key] = value.astype(fnp.uint64)
        return widened[key]

    # A run of consecutive constant-only words goes out as one broadcast, not a
    # `full` plus a stack operand each. The b-stream is where this pays: its
    # lin-row masks are constants, so sha2's carries 156 constant-only words in
    # four runs, against z/a's 21 in one (the zero pad).
    blocks: list = []
    const_run: list[int] = []
    array_run: list = []

    def flush_const():
        if const_run:
            row = fnp.asarray(np.array(const_run, dtype=np.uint64))
            blocks.append(fnp.broadcast_to(row, (n, len(const_run))))
            const_run.clear()

    def flush_array():
        if array_run:
            blocks.append(fnp.stack(array_run, axis=1))
            array_run.clear()

    for w in range(words):
        cw = (const_acc >> (64 * w)) & _ONES64
        acc = None
        for value, shift, is_high in contrib[w]:
            v = as_u64(value)
            v = (v >> fnp.uint64(shift)) if is_high else (v << fnp.uint64(shift))
            acc = v if acc is None else acc | v
        if acc is None:
            flush_array()
            const_run.append(cw)
        else:
            flush_const()
            array_run.append((acc | fnp.uint64(cw)) if cw else acc)
    flush_const()
    flush_array()
    return fnp.concatenate(blocks, axis=1)
