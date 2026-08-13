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

The R1CS is `a AND b = z` per bit, in the row forms `common` documents.
Only `b_new`/`d_new` need lin rows: `a_2`/`c_2` are already pinned as ADD-row
sums.

Emission is `common.emit`, shared with sha2: each of the 256 output words
is assembled directly from the u32 field values, every bit offset a
compile-time constant, so a field lands in one word (or straddles two) with
static shifts and all-constant regions (the b-stream's lin masks, the constant
wire, the padding) fold into one host-side K-bit literal per stream. Nothing
per-bit is ever materialized, which is what lets a 2^18 batch emit at output
size instead of 8x it. The streaming-writer word tricks in the Rust source (the
shift-by-one from the constant wire, the flags>>31 pending bit) are artifacts
of its sequential emission and fall out of the same offsets.
"""

from __future__ import annotations

import functools

import frx
import frx.numpy as fnp
import numpy as np
from frx import lax
from hash_frx.blake3.compress import IV, MSG_PERMUTATION, ROUNDS
from hash_frx.word import rotr

from flock_zorch.r1cs_hashes.common import (
    CARRY_BITS,
    ONES32,
    WORD_BITS,
    add_row,
    emit,
)

K_LOG = 14
K = 1 << K_LOG
WORDS_PER_BLOCK = K // 64  # 256 u64 per block per stream

CV_WORDS = 8  # the chaining value, and each half of the output
M_WORDS = 16
N_G = ROUNDS * 8
G_STRIDE = 6 * CARRY_BITS + 2 * WORD_BITS  # 250

CV_BASE = 0
OUT_LO_BASE = CV_BASE + CV_WORDS * WORD_BITS  # 256
Z_CONST_POS = OUT_LO_BASE + CV_WORDS * WORD_BITS  # 512
M_BASE = Z_CONST_POS + 1  # 513
COUNTER_BASE = M_BASE + M_WORDS * WORD_BITS  # 1,025
# counter_lo, counter_hi, block_len, flags.
G_BASE = COUNTER_BASE + 4 * WORD_BITS  # 1,153
OUT_HI_BASE = G_BASE + N_G * G_STRIDE  # 15,153
USEFUL_BITS = OUT_HI_BASE + CV_WORDS * WORD_BITS  # 15,409

assert (Z_CONST_POS, G_BASE, OUT_HI_BASE, USEFUL_BITS) == (512, 1153, 15153, 15409)

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


def witness_blake3(cv, m, counter, block_len, flags):
    """Packed z/a/b witness streams for a batch of BLAKE3 compressions.

    cv: uint32 [N, 8], m: uint32 [N, 16], counter: uint64 [N],
    block_len/flags: uint32 [N] -> three uint64 [N, 256] arrays (z, a, b).

    GPU runs `blake3_witness_pallas`'s fused kernel, which walks the chain once
    for all three streams and stores each output word as it is finished; this
    XLA expression stays as the portable path and as the kernel's oracle, which
    is how `zorch.utils.binary_field.byte_select_xor_reduce` splits the same way.
    The kernel is Triton, so it has no CPU lowering at all ("Only interpret
    mode is supported on CPU backend") — the split is forced, not a tuning
    choice. `blake3_witness_test` byte-compares the two wherever both can run.
    """
    if frx.default_backend() != "gpu":
        return _witness_blake3_xla(cv, m, counter, block_len, flags)
    # Deferred: blake3_witness_pallas reads this module's layout constants, so a
    # module-scope import here would be a cycle.
    from flock_zorch.r1cs_hashes import blake3_witness_pallas

    return blake3_witness_pallas.witness_blake3(cv, m, counter, block_len, flags)


@frx.jit
def _witness_blake3_xla(cv, m, counter, block_len, flags):
    """The portable emission: every output word assembled in one expression.

    Kept as the CPU path and as the kernel's independent reference. It is also
    the slower one on GPU by 2.6x — one fusion per stream has to keep the
    sequential 336-row chain live across all 256 words, which pins it at the
    255-register cap (fractalyze/xla#495).
    """
    n = cv.shape[0]
    t_lo = counter.astype(fnp.uint32)
    t_hi = (counter >> fnp.uint64(32)).astype(fnp.uint32)

    state = [cv[:, i] for i in range(CV_WORDS)]
    state += [fnp.full((n,), c, dtype=fnp.uint32) for c in IV[:4]]
    state += [t_lo, t_hi, block_len, flags]

    # The three streams are appended in lockstep so they cannot diverge; the
    # b-stream's lin rows are int constants, so they fold into `emit`'s K-bit
    # literal instead of costing device work.
    zf, af, bf = [], [], []

    def put(off, width, zv, av, bv):
        zf.append((off, width, zv))
        af.append((off, width, av))
        bf.append((off, width, bv))

    def put_lin(off, v):
        """A wire pinned by identity, against the all-ones mask."""
        put(off, WORD_BITS, v, v, ONES32)

    def add(off, x, y):
        """An ADD with its carry row; returns the mod-2^32 sum."""
        s, left, right, carry = add_row(x, y)
        put(off, CARRY_BITS, carry, left, right)
        return s

    for w in range(CV_WORDS):
        put_lin(CV_BASE + WORD_BITS * w, cv[:, w])
    put(Z_CONST_POS, 1, 1, 1, 1)
    for i in range(M_WORDS):
        put_lin(M_BASE + WORD_BITS * i, m[:, i])
    for i, v in enumerate((t_lo, t_hi, block_len, flags)):
        put_lin(COUNTER_BASE + WORD_BITS * i, v)

    for r in range(ROUNDS):
        sched = _SCHEDULE[r]
        for g in range(8):
            la, lb, lc, ld = _G_LANES[g]
            mx = m[:, sched[2 * g]]
            my = m[:, sched[2 * g + 1]]
            a0, b0, c0, d0 = state[la], state[lb], state[lc], state[ld]

            base = G_BASE + (r * 8 + g) * G_STRIDE
            slot = lambda i: base + i * CARRY_BITS  # noqa: E731
            a1 = add(slot(1), add(slot(0), a0, b0), mx)
            d1 = rotr(d0 ^ a1, 16)
            c1 = add(slot(2), c0, d1)
            b1 = rotr(b0 ^ c1, 12)
            a2 = add(slot(4), add(slot(3), a1, b1), my)
            d2 = rotr(d1 ^ a2, 8)
            c2 = add(slot(5), c1, d2)
            b_new = rotr(b1 ^ c2, 7)
            lin = slot(6)  # the two lin rows close the G block
            put_lin(lin, b_new)
            put_lin(lin + WORD_BITS, d2)

            state[la], state[lb], state[lc], state[ld] = a2, b_new, c2, d2

    # out_lo is the field out of bit order: it sits at bit 256 but is only known
    # after the last round, which is exactly what the offset form absorbs.
    for w in range(CV_WORDS):
        put_lin(OUT_LO_BASE + WORD_BITS * w, state[w] ^ state[w + 8])
        put_lin(OUT_HI_BASE + WORD_BITS * w, state[w + 8] ^ cv[:, w])

    def pack(fields):
        return emit(fields, n, words=WORDS_PER_BLOCK, useful_bits=USEFUL_BITS)

    return pack(zf), pack(af), pack(bf)


_SPLITMIX_GOLDEN = 0x9E37_79B9_7F4A_7C15


@functools.partial(frx.jit, static_argnums=(1,))
def blocks_from_seed(seed, log2_size: int):
    """The flock-challenge benchmark's `generate_compressions`, on device.

    seed: uint64 scalar -> (cv, m, counter, block_len, flags) for 2^log2_size
    compressions, bit-identical to `flock_benchmark_common`'s host generator
    (splitmix64, 25 u32 draws per compression: 8 cv + 16 message + 1 counter;
    block_len/flags fixed at 64/11). splitmix's state advances by a constant
    per draw, so draw j is `mix(s0 + (j+1)*GOLDEN)` — no sequential state,
    every draw independent. With this, only the 8-byte seed crosses the host
    boundary: blocks, witness, and prove are all device-resident.

    `log2_size` is static (the arange shape depends on it); the seed is
    traced, so per-trial seeds reuse the compiled program.
    """
    n = 1 << log2_size
    golden = fnp.uint64(_SPLITMIX_GOLDEN)
    # u64::from(log2_size).rotate_left(29) — log2_size < 2^35, so no wrap.
    s0 = fnp.asarray(seed, dtype=fnp.uint64) ^ fnp.uint64(log2_size << 29)
    j = fnp.arange(1, n * 25 + 1, dtype=fnp.uint64)
    z = s0 + j * golden
    z = (z ^ (z >> fnp.uint64(30))) * fnp.uint64(0xBF58_476D_1CE4_E5B9)
    z = (z ^ (z >> fnp.uint64(27))) * fnp.uint64(0x94D0_49BB_1331_11EB)
    draws = (z ^ (z >> fnp.uint64(31))).astype(fnp.uint32).reshape(n, 25)
    cv = draws[:, 0:8]
    m = draws[:, 8:24]
    counter = draws[:, 24].astype(fnp.uint64)
    block_len = fnp.full((n,), 64, dtype=fnp.uint32)
    flags = fnp.full((n,), 11, dtype=fnp.uint32)
    return cv, m, counter, block_len, flags


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


def lincheck_stripe(z):
    """The lincheck byte stripe from the packed z stream.

    z: uint64 [N, 256] with N divisible by 8 -> uint8 [N/8, 16384].

    GPU runs `blake3_witness_pallas`'s kernel where the batch allows it; this
    XLA expression is the portable path and the kernel's oracle, the same split
    `witness_blake3` makes. Batches that are not a whole number of the kernel's
    tile fall through to it as well — real proves are powers of two, and
    padding a stripe would mean inventing blocks.
    """
    from flock_zorch.r1cs_hashes import blake3_witness_pallas

    if (
        frx.default_backend() == "gpu"
        and z.shape[0] % blake3_witness_pallas.stripe_rows() == 0
    ):
        return blake3_witness_pallas.lincheck_stripe(z)
    return _lincheck_stripe_xla(z)


@frx.jit
def _lincheck_stripe_xla(z):
    """The lincheck byte stripe from the packed z stream, on device.

    z: uint64 [N, 256] with N divisible by 8 -> uint8 [N/8, 16384].
    Byte `g*16384 + i*64 + c*8 + t` holds, in bit r, bit `i*64 + c*8 + t`
    of block `8g + r`'s z — flock's stripe layout, one byte per bit column
    covering all 8 blocks of the group. The device twin of the host port
    `flock_zorch.pcs.pack.pack_z_lincheck_from_packed`; blake3_witness_test
    pins the two byte-identical.
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
