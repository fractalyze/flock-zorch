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
import numpy as np
from frx import lax
from hash_frx.blake3.compress import IV, MSG_PERMUTATION, ROUNDS
from hash_frx.word import rotr

K_LOG = 14
K = 1 << K_LOG
WORDS_PER_BLOCK = K // 64  # 256 u64 per block per stream

N_G = ROUNDS * 8
CARRY_BITS = 31
G_STRIDE = 6 * CARRY_BITS + 2 * 32  # 250
USEFUL_BITS = 2 * 256 + 1 + 16 * 32 + 4 * 32 + N_G * G_STRIDE + 8 * 32  # 15,409
PAD_BITS = K - USEFUL_BITS  # 975

# The lincheck stripe transposes whole words, so it covers ceil(USEFUL_BITS/64)
# words = 15,424 bit columns; the remaining 960 bytes per group are the honest
# zero pad flock's `cargo test` contract requires (the production fold never
# reads them, and release flock leaves them unwritten).
STRIPE_USEFUL_WORDS = -(-USEFUL_BITS // 64)  # 241
STRIPE_BYTES_PER_GROUP = K  # one byte per bit column, 8 blocks per byte

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

# Message index per (round, G): the per-round permutation pre-applied so round r
# reads the ORIGINAL message — the same composition flock-prover unrolls into
# literals. G `g` consumes entries 2g (mx) and 2g+1 (my).
_SCHEDULE = [tuple(range(16))]
for _ in range(ROUNDS - 1):
    _SCHEDULE.append(tuple(_SCHEDULE[-1][i] for i in MSG_PERMUTATION))

_MASK31 = fnp.uint32(0x7FFF_FFFF)


def _add_row(x, y):
    """One 32-bit ADD with its R1CS row parts: (sum, left, right, carry)."""
    s = x + y
    cin = s ^ x ^ y
    left = (x ^ cin) & _MASK31
    right = (y ^ cin) & _MASK31
    return s, left, right, left & right


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
        cw = (const_acc >> (64 * w)) & ((1 << 64) - 1)
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


def witness_blake3(cv, m, counter, block_len, flags):
    """Packed z/a/b witness streams for a batch of BLAKE3 compressions.

    cv: uint32 [N, 8], m: uint32 [N, 16], counter: uint64 [N],
    block_len/flags: uint32 [N] -> three uint64 [N, 256] arrays (z, a, b).

    GPU runs `witgen_pallas`'s fused kernel, which walks the chain once for all
    three streams and stores each output word as it is finished; this XLA
    expression stays as the portable path and as the kernel's oracle, which is
    how `zorch.utils.binary_field.byte_select_xor_reduce` splits the same way.
    The kernel is Triton, so it has no CPU lowering at all ("Only interpret
    mode is supported on CPU backend") — the split is forced, not a tuning
    choice. `witgen_test` byte-compares the two wherever both can run.
    """
    if frx.default_backend() != "gpu":
        return _witness_blake3_xla(cv, m, counter, block_len, flags)
    # Deferred: witgen_pallas reads this module's layout constants, so a
    # module-scope import here would be a cycle.
    from flock_zorch import witgen_pallas

    return witgen_pallas.witness_blake3(cv, m, counter, block_len, flags)


@frx.jit
def _witness_blake3_xla(cv, m, counter, block_len, flags):
    """The portable emission: assemble every output word as one stacked stream.

    Kept as the CPU path and as the kernel's independent reference. It is also
    the slower one on GPU by 2.6x — one fusion per stream has to keep the
    sequential 336-row chain live across all 256 words, which pins it at the
    255-register cap (fractalyze/xla#495).
    """
    n = cv.shape[0]
    t_lo = counter.astype(fnp.uint32)
    t_hi = (counter >> fnp.uint64(32)).astype(fnp.uint32)

    state = [cv[:, i] for i in range(8)]
    state += [fnp.full((n,), c, dtype=fnp.uint32) for c in IV[:4]]
    state += [t_lo, t_hi, block_len, flags]

    # The three streams are appended in lockstep so they cannot diverge; the
    # b-stream's lin rows are int constants, so they fold into _emit's K-bit
    # literal instead of costing device work.
    zg, ag, bg = [], [], []

    def put(zv, av, bv, width):
        zg.append((zv, width))
        ag.append((av, width))
        bg.append((bv, width))

    def add(x, y):
        s, left, right, carry = _add_row(x, y)
        put(carry, left, right, CARRY_BITS)
        return s

    for r in range(ROUNDS):
        sched = _SCHEDULE[r]
        for g in range(8):
            la, lb, lc, ld = _G_LANES[g]
            mx = m[:, sched[2 * g]]
            my = m[:, sched[2 * g + 1]]
            a0, b0, c0, d0 = state[la], state[lb], state[lc], state[ld]

            a1 = add(add(a0, b0), mx)
            d1 = rotr(d0 ^ a1, 16)
            c1 = add(c0, d1)
            b1 = rotr(b0 ^ c1, 12)
            a2 = add(add(a1, b1), my)
            d2 = rotr(d1 ^ a2, 8)
            c2 = add(c1, d2)
            b_new = rotr(b1 ^ c2, 7)
            for v in (b_new, d2):
                put(v, v, 0xFFFF_FFFF, 32)

            state[la], state[lb], state[lc], state[ld] = a2, b_new, c2, d2

    out_lo = [state[w] ^ state[w + 8] for w in range(8)]
    out_hi = [state[w + 8] ^ cv[:, w] for w in range(8)]

    head = [(cv[:, i], 32) for i in range(8)]
    head += [(v, 32) for v in out_lo]
    head += [(1, 1)]
    head += [(m[:, i], 32) for i in range(16)]
    head += [(t_lo, 32), (t_hi, 32), (block_len, 32), (flags, 32)]
    tail = [(v, 32) for v in out_hi]
    pad = [(0, PAD_BITS)]

    def ones(fields):
        # b carries the lin-row all-ones mask exactly where z/a carry values.
        return [((1 << w) - 1, w) for _, w in fields]

    z = _emit(head + zg + tail + pad, n)
    a = _emit(head + ag + tail + pad, n)
    b = _emit(ones(head) + bg + ones(tail) + pad, n)
    return z, a, b


def extract_inputs(z_lanes):
    """Recover every block's compression inputs from a packed z stream.

    z_lanes: host uint64 [N*128, 2] (the golden loader's lane form) ->
    (cv, m, counter, block_len, flags) exactly as `witness_blake3` takes
    them. Inverts the prefix packing: the constant-1 wire at bit 512 shifts
    words 8..17 up by one bit, so each u64 holds 1+32+31 bits and every odd
    value's top bit sits in the following word (flags' in word 18). Lets a
    golden gate regenerate its own witness with no extra fixture.
    """
    zw = np.asarray(z_lanes, dtype=np.uint64).reshape(-1, WORDS_PER_BLOCK)
    n = zw.shape[0]
    if not np.all(zw[:, 8] & 1):
        raise ValueError("constant-1 wire missing — not a packed blake3 z stream")
    cv = np.zeros((n, 8), np.uint32)
    for i in range(4):
        w = zw[:, i]
        cv[:, 2 * i] = w.astype(np.uint32)
        cv[:, 2 * i + 1] = (w >> np.uint64(32)).astype(np.uint32)
    vals = np.zeros((n, 20), np.uint32)
    for i in range(10):
        w = zw[:, 8 + i]
        nxt = zw[:, 9 + i]
        vals[:, 2 * i] = (w >> np.uint64(1)).astype(np.uint32)
        vals[:, 2 * i + 1] = (
            (w >> np.uint64(33)) | ((nxt & np.uint64(1)) << np.uint64(31))
        ).astype(np.uint32)
    counter = vals[:, 16].astype(np.uint64) | (vals[:, 17].astype(np.uint64) << 32)
    return cv, vals[:, :16].copy(), counter, vals[:, 18].copy(), vals[:, 19].copy()


def _transpose_8x8_bits(x):
    """Hacker's Delight 7-3: transpose the 8x8 bit matrix held in each u64
    (byte i = row i); bit r*8+c of the input lands at bit c*8+r."""
    t = (x ^ (x >> fnp.uint64(7))) & fnp.uint64(0x00AA_00AA_00AA_00AA)
    x = x ^ t ^ (t << fnp.uint64(7))
    t = (x ^ (x >> fnp.uint64(14))) & fnp.uint64(0x0000_CCCC_0000_CCCC)
    x = x ^ t ^ (t << fnp.uint64(14))
    t = (x ^ (x >> fnp.uint64(28))) & fnp.uint64(0x0000_0000_F0F0_F0F0)
    x = x ^ t ^ (t << fnp.uint64(28))
    return x


@frx.jit
def lincheck_stripe(z):
    """The lincheck byte stripe from the packed z stream, on device.

    z: uint64 [N, 256] with N divisible by 8 -> uint8 [N/8, 16384].
    Byte `g*16384 + i*64 + c*8 + t` holds, in bit r, bit `i*64 + c*8 + t`
    of block `8g + r`'s z — flock's stripe layout, one byte per bit column
    covering all 8 blocks of the group. The device twin of the host port
    `flock_zorch.pcs.pack.pack_z_lincheck_from_packed`; witgen_test pins
    the two byte-identical.
    """
    n = z.shape[0]
    if n % 8:
        raise ValueError(f"stripe needs a multiple of 8 blocks, got {n}")
    grp = z.reshape(n // 8, 8, WORDS_PER_BLOCK)[:, :, :STRIPE_USEFUL_WORDS]

    cols = []
    for c in range(8):
        packed = None
        for r in range(8):
            byte = (grp[:, r, :] >> fnp.uint64(8 * c)) & fnp.uint64(0xFF)
            term = byte << fnp.uint64(8 * r)
            packed = term if packed is None else packed | term
        cols.append(_transpose_8x8_bits(packed))

    out = fnp.stack(cols, axis=-1)  # [G, 241, 8]; u64 bytes = columns c*8..c*8+8
    out8 = lax.bitcast_convert_type(out, fnp.uint8).reshape(
        n // 8, STRIPE_USEFUL_WORDS * 64
    )
    pad = fnp.zeros(
        (n // 8, STRIPE_BYTES_PER_GROUP - STRIPE_USEFUL_WORDS * 64), dtype=fnp.uint8
    )
    return fnp.concatenate([out8, pad], axis=1)
