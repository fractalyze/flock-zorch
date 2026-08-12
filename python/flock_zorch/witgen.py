# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Device-side flock BLAKE3 R1CS witness generation.

Emits the packed `z`/`a`/`b` bit-streams for a batch of BLAKE3 compressions,
bit-identical to flock-prover's `build_block_witness_ab_stream_into` (the
generator behind `generate_witness_with_ab_packed_and_lincheck`). The prove
already consumes exactly these buffers; producing them on device removes the
host generation pass and its 1.5 GiB H2D crossing per m32 prove.

Layout (bit indices into each block's 2^14-bit slice, LSB-first within u64
words; every offset is a compile-time constant):

    [    0,   256)  input chaining value cv[0..8]
    [  256,   512)  out_lo[w] = state[w] ^ state[w+8]   (the output cv)
    [  512]         the constant-1 wire
    [  513,  1025)  message m[0..16]
    [ 1025,  1153)  counter_lo, counter_hi, block_len, flags
    [ 1153, 15153)  56 G blocks x 250 bits:
                      6 ADD rows x 31 carry bits, then b_new, d_new (32 each)
    [15153, 15409)  out_hi[w] = state[w+8] ^ cv[w]
    [15409, 16384)  zero padding

The R1CS is `a AND b = z` per bit. An ADD row for `sum = x + y` (mod 2^32)
stores `carry = left AND right` where `left = (x ^ cin) & 0x7FFF_FFFF`,
`right = (y ^ cin) & 0x7FFF_FFFF`, `cin = sum ^ x ^ y` — bit 31 is the
discarded mod-2^32 carry-out and gets no slot. A lin-id row pins a wire `v`
via `z = a = v`, `b = 0xFFFF_FFFF`. Only `b_new`/`d_new` need lin rows:
`a_2`/`c_2` are already pinned as ADD-row sums.

Emission assembles each of the 256 output words directly from the u32 field
values: every field's bit offset is a compile-time constant, so a field
lands in one word (or straddles two) with static shifts, and all-constant
regions (the b-stream's lin masks, the constant wire, the padding) fold
into one host-side K-bit literal per stream. Nothing per-bit is ever
materialized, which is what lets a 2^18 batch emit at output size instead
of 8x it. The streaming-writer word tricks in the Rust source (the
shift-by-one from the constant wire, the flags>>31 pending bit) are
artifacts of its sequential emission and fall out of the same offsets.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp

K_LOG = 14
K = 1 << K_LOG
WORDS_PER_BLOCK = K // 64  # 256 u64 per block per stream

N_ROUNDS = 7
N_G = N_ROUNDS * 8
CARRY_BITS = 31
G_STRIDE = 6 * CARRY_BITS + 2 * 32  # 250
USEFUL_BITS = 2 * 256 + 1 + 16 * 32 + 4 * 32 + N_G * G_STRIDE + 8 * 32  # 15,409
PAD_BITS = K - USEFUL_BITS  # 975

_IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A)

_G_LANES = (
    (0, 4, 8, 12),
    (1, 5, 9, 13),
    (2, 6, 10, 14),
    (3, 7, 11, 15),
    (0, 5, 10, 15),
    (1, 6, 11, 12),
    (2, 7, 8, 13),
    (3, 4, 9, 14),
)

# Message index per (round, G): the per-round permutation pre-applied, exactly
# as flock-prover unrolls it. G `g` consumes entries 2g (mx) and 2g+1 (my).
_SCHEDULE = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8),
    (3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1),
    (10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6),
    (12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4),
    (9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7),
    (11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13),
)

_MASK31 = fnp.uint32(0x7FFF_FFFF)


def _ror(x, n: int):
    return (x >> fnp.uint32(n)) | (x << fnp.uint32(32 - n))


def _add_row(x, y):
    """One 32-bit ADD with its R1CS row parts: (sum, left, right, carry)."""
    s = x + y
    cin = s ^ x ^ y
    left = (x ^ cin) & _MASK31
    right = (y ^ cin) & _MASK31
    return s, left, right, left & right


_M64 = (1 << 64) - 1


def _emit(fields, n: int):
    """Assemble one packed stream: [(value, width)] in bit order -> u64 [N, 256].

    `value` is a uint32 [N] array (pre-masked to `width` bits, as the row
    builders guarantee) or a Python int constant of any width. Array fields
    contribute to one output word, or two when they straddle a boundary,
    with shifts fixed at trace time; int fields accumulate into a single
    K-bit literal that seeds the constant words.
    """
    const_acc = 0
    contrib: list[list[tuple]] = [[] for _ in range(WORDS_PER_BLOCK)]
    pos = 0
    for value, width in fields:
        if isinstance(value, int):
            const_acc |= (value & ((1 << width) - 1)) << pos
        else:
            w, s = divmod(pos, 64)
            contrib[w].append((value, s, False))
            if s + width > 64:
                contrib[w + 1].append((value, 64 - s, True))
        pos += width
    assert pos == K, pos

    cols = []
    for w in range(WORDS_PER_BLOCK):
        cw = (const_acc >> (64 * w)) & _M64
        acc = None
        for value, shift, is_high in contrib[w]:
            v = value.astype(fnp.uint64)
            v = (v >> fnp.uint64(shift)) if is_high else (v << fnp.uint64(shift))
            acc = v if acc is None else acc | v
        if acc is None:
            cols.append(fnp.full((n,), cw, dtype=fnp.uint64))
        else:
            cols.append(acc | fnp.uint64(cw) if cw else acc)
    return fnp.stack(cols, axis=1)


@frx.jit
def witness_blake3(cv, m, counter, block_len, flags):
    """Packed z/a/b witness streams for a batch of BLAKE3 compressions.

    cv: uint32 [N, 8], m: uint32 [N, 16], counter: uint64 [N],
    block_len/flags: uint32 [N] -> three uint64 [N, 256] arrays (z, a, b).
    """
    n = cv.shape[0]
    t_lo = counter.astype(fnp.uint32)
    t_hi = (counter >> fnp.uint64(32)).astype(fnp.uint32)

    state = [cv[:, i] for i in range(8)]
    state += [fnp.full((n,), c, dtype=fnp.uint32) for c in _IV]
    state += [t_lo, t_hi, block_len, flags]

    # Per-G row values in emission order; (value, nbits) per field. The
    # b-stream's lin rows are int constants, so they fold into _emit's
    # K-bit literal instead of costing device work.
    zg, ag, bg = [], [], []
    for r in range(N_ROUNDS):
        sched = _SCHEDULE[r]
        for g in range(8):
            la, lb, lc, ld = _G_LANES[g]
            mx = m[:, sched[2 * g]]
            my = m[:, sched[2 * g + 1]]
            a0, b0, c0, d0 = state[la], state[lb], state[lc], state[ld]

            tmp0, l0, r0, cy0 = _add_row(a0, b0)
            a1, l1, r1, cy1 = _add_row(tmp0, mx)
            d1 = _ror(d0 ^ a1, 16)
            c1, l2, r2, cy2 = _add_row(c0, d1)
            b1 = _ror(b0 ^ c1, 12)
            tmp1, l3, r3, cy3 = _add_row(a1, b1)
            a2, l4, r4, cy4 = _add_row(tmp1, my)
            d2 = _ror(d1 ^ a2, 8)
            c2, l5, r5, cy5 = _add_row(c1, d2)
            b_new = _ror(b1 ^ c2, 7)

            zg += [(cy0, 31), (cy1, 31), (cy2, 31), (cy3, 31), (cy4, 31), (cy5, 31)]
            ag += [(l0, 31), (l1, 31), (l2, 31), (l3, 31), (l4, 31), (l5, 31)]
            bg += [(r0, 31), (r1, 31), (r2, 31), (r3, 31), (r4, 31), (r5, 31)]
            zg += [(b_new, 32), (d2, 32)]
            ag += [(b_new, 32), (d2, 32)]
            bg += [(0xFFFF_FFFF, 32), (0xFFFF_FFFF, 32)]

            state[la], state[lb], state[lc], state[ld] = a2, b_new, c2, d2

    out_lo = [state[w] ^ state[w + 8] for w in range(8)]
    out_hi = [state[w + 8] ^ cv[:, w] for w in range(8)]

    # z and a share every non-G field; b is all-ones through bit 1153 (its
    # prefix lin rows) and over out_hi, zero over the padding.
    head = [(cv[:, i], 32) for i in range(8)]
    head += [(v, 32) for v in out_lo]
    head += [(1, 1)]
    head += [(m[:, i], 32) for i in range(16)]
    head += [(t_lo, 32), (t_hi, 32), (block_len, 32), (flags, 32)]
    tail = [(v, 32) for v in out_hi]
    pad = [(0, PAD_BITS)]

    z = _emit(head + zg + tail + pad, n)
    a = _emit(head + ag + tail + pad, n)
    b = _emit([((1 << 1153) - 1, 1153)] + bg + [((1 << 256) - 1, 256)] + pad, n)
    return z, a, b
