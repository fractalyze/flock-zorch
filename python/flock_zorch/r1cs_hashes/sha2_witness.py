# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Device-side flock SHA-256 R1CS witness generation.

Emits the packed `z`/`a`/`b` bit-streams for a batch of SHA-256 compressions,
bit-identical to flock-prover's `r1cs_hashes::sha2` witness builder, the way
`blake3_witness` already does for BLAKE3 and `keccak_witness` for the keccak
family.

sha2 is the bridge between the two families: it packs tightly like BLAKE3, so
fields straddle 64-bit word boundaries, but it also materializes Ch and Maj as
AND-output rows like keccak's chi. It is the only circuit of the four that
carries both row forms.

Layout (bit indices into each block's 2^15-bit slice, LSB-first within u64
words; every offset is a compile-time constant):

    [    0,   256)  H_in[w]                     8 x 32   lin-id
    [  256,   512)  H_out[w]                    8 x 32   lin-id
    [  512,  1024)  M_in[i]                    16 x 32   lin-id
    [ 1024,  3072)  ch_and[r]                  64 x 32   AND
    [ 3072,  5120)  maj_and[r]                 64 x 32   AND
    [ 5120, 19008)  round carries         64 x 7 x 31    ADD carry
    [19008, 20544)  W[t]                       48 x 32   lin-id
    [20544, 25008)  schedule carries      48 x 3 x 31    ADD carry
    [25008, 27056)  T1[r]                      64 x 32   lin-id
    [27056, 29104)  E_NEW[r]                   64 x 32   lin-id
    [29104, 31152)  A_NEW[r]                   64 x 32   lin-id
    [31152, 31400)  output carries              8 x 31   ADD carry
    [31400]         the constant-1 wire
    [31401, 32768)  zero padding

Two facts about this layout are worth stating because the module's own `//!`
header in flock disagrees with its constants (see docs/development.md).
`Z_CONST` sits at
the END, at bit 31,400, not at bit 512 — which is what keeps the first 1,024
bits four clean 256-bit slots the Merkle-path protocol can address with
single-bit selectors. And because the constant wire is not mid-block, nothing
here needs BLAKE3's `extract_inputs` one-bit unshift: `H_in` and `M_in` are
word-aligned and read straight out of the packed z.

The R1CS is `a AND b = z` per bit, in the row forms `common` documents;
sha2 is the circuit that uses all three.

Emission is `common.emit`, shared with BLAKE3, and sha2 is the reason it
addresses fields by offset rather than walking a bit cursor: sha2's computation
order is not its bit order, with `ch_and` for every round living at bits
[1024, 3072) but produced interleaved with that round's carries at 5,120 and
up. Each field therefore carries its own offset and lands with shifts fixed at
trace time.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from hash_frx.word import rotr

from flock_zorch.r1cs_hashes.common import (
    CARRY_BITS,
    ONES32,
    WORD_BITS,
    add_row,
    emit,
)

K_LOG = 15
K = 1 << K_LOG
WORDS_PER_BLOCK = K // 64  # 512 u64 per block per stream

SLOT_BITS = 256
H_WORDS = 8
M_WORDS = 16
N_ROUNDS = 64
N_SCHED = 48
ADDS_PER_ROUND = 7
ADDS_PER_SCHED = 3

# The first four 256-bit slots are addressable with single-bit selectors, which
# is why the constant wire was moved to the end of the block rather than left
# between them.
H_BASE = 0
H_OUT_BASE = SLOT_BITS  # 256
M_BASE = 2 * SLOT_BITS  # 512
CH_AND_BASE = M_BASE + M_WORDS * WORD_BITS  # 1,024
MAJ_AND_BASE = CH_AND_BASE + N_ROUNDS * WORD_BITS  # 3,072
ROUND_CARRY_BASE = MAJ_AND_BASE + N_ROUNDS * WORD_BITS  # 5,120
ROUND_CARRY_STRIDE = ADDS_PER_ROUND * CARRY_BITS  # 217
W_BASE = ROUND_CARRY_BASE + N_ROUNDS * ROUND_CARRY_STRIDE  # 19,008
SCHED_CARRY_BASE = W_BASE + N_SCHED * WORD_BITS  # 20,544
SCHED_CARRY_STRIDE = ADDS_PER_SCHED * CARRY_BITS  # 93
T1_BASE = SCHED_CARRY_BASE + N_SCHED * SCHED_CARRY_STRIDE  # 25,008
E_NEW_BASE = T1_BASE + N_ROUNDS * WORD_BITS  # 27,056
A_NEW_BASE = E_NEW_BASE + N_ROUNDS * WORD_BITS  # 29,104
OUT_CARRY_BASE = A_NEW_BASE + N_ROUNDS * WORD_BITS  # 31,152
Z_CONST_POS = OUT_CARRY_BASE + H_WORDS * CARRY_BITS  # 31,400
USEFUL_BITS = Z_CONST_POS + 1  # 31,401

assert (W_BASE, T1_BASE, Z_CONST_POS) == (19008, 25008, 31400)
assert USEFUL_BITS == 31401

# FIPS 180-4 section 4.2.2. Transcribed rather than imported: hash-frx keeps its
# copy private, and the test anchors H_out against its compression, which fails
# loudly on a typo here.
# fmt: off
_K = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)
# fmt: on
assert len(_K) == N_ROUNDS


@frx.jit
def witness_sha2(h_in, m):
    """Packed z/a/b witness streams for a batch of SHA-256 compressions.

    h_in: uint32 [N, 8] (the input chaining value, a..h), m: uint32 [N, 16] ->
    three uint64 [N, 512] arrays (z, a, b).

    The three streams are appended in lockstep so they cannot diverge; the
    b-stream's lin rows are int constants, so they fold into `emit`'s K-bit
    literal instead of costing device work.
    """
    n = h_in.shape[0]
    zf, af, bf = [], [], []

    def put(off, width, zv, av, bv):
        zf.append((off, width, zv))
        af.append((off, width, av))
        bf.append((off, width, bv))

    def put_lin(off, v):
        """A wire pinned by identity, against the all-ones mask."""
        put(off, WORD_BITS, v, v, ONES32)

    def put_and(off, x, y):
        """An AND-output row: Ch and Maj are materialized, not folded.

        Returns the AND so the caller can finish the Ch/Maj XOR without
        building it a second time.
        """
        z = x & y
        put(off, WORD_BITS, z, x, y)
        return z

    def add(off, x, y):
        """An ADD with its carry row; returns the mod-2^32 sum."""
        s, left, right, carry = add_row(x, y)
        put(off, CARRY_BITS, carry, left, right)
        return s

    for w in range(H_WORDS):
        put_lin(H_BASE + WORD_BITS * w, h_in[:, w])
    for i in range(M_WORDS):
        put_lin(M_BASE + WORD_BITS * i, m[:, i])

    # Message schedule: W[16..64], three carries per t then the W[t] lin row.
    w_sched = [m[:, i] for i in range(M_WORDS)]
    for t in range(M_WORDS, N_ROUNDS):
        j = t - M_WORDS
        x2, x7, x15, x16 = (w_sched[t - k] for k in (2, 7, 15, 16))
        s1 = rotr(x2, 17) ^ rotr(x2, 19) ^ (x2 >> fnp.uint32(10))
        s0 = rotr(x15, 7) ^ rotr(x15, 18) ^ (x15 >> fnp.uint32(3))
        base = SCHED_CARRY_BASE + j * SCHED_CARRY_STRIDE
        acc = s1
        for i, operand in enumerate((x7, s0, x16)):
            acc = add(base + i * CARRY_BITS, acc, operand)
        put_lin(W_BASE + WORD_BITS * j, acc)
        w_sched.append(acc)

    st = [h_in[:, i] for i in range(H_WORDS)]
    for r in range(N_ROUNDS):
        aa, bb, cc, dd, ee, ff, gg, hh = st

        # Ch(e, f, g) = g ^ (e & (f ^ g)) and Maj(a, b, c) = a ^ ((b^a) & (c^a)):
        # the AND is the materialized row, the XOR is free.
        f_xor_g = ff ^ gg
        ch_out = put_and(CH_AND_BASE + WORD_BITS * r, ee, f_xor_g) ^ gg

        b_xor_a, c_xor_a = bb ^ aa, cc ^ aa
        maj_out = put_and(MAJ_AND_BASE + WORD_BITS * r, b_xor_a, c_xor_a) ^ aa

        s1e = rotr(ee, 6) ^ rotr(ee, 11) ^ rotr(ee, 25)
        s0a = rotr(aa, 2) ^ rotr(aa, 13) ^ rotr(aa, 22)
        # A scalar, not a broadcast column: the round constant only ever feeds
        # the ADD arithmetic, never `put`, so it needs no batch dimension.
        kr = fnp.uint32(_K[r])

        base = ROUND_CARRY_BASE + r * ROUND_CARRY_STRIDE
        slot = lambda i: base + i * CARRY_BITS  # noqa: E731
        t1 = add(slot(0), hh, s1e)
        t1 = add(slot(1), t1, ch_out)
        t1 = add(slot(2), t1, kr)
        t1 = add(slot(3), t1, w_sched[r])
        t2 = add(slot(4), s0a, maj_out)
        e_new = add(slot(5), dd, t1)
        a_new = add(slot(6), t1, t2)

        put_lin(T1_BASE + WORD_BITS * r, t1)
        put_lin(E_NEW_BASE + WORD_BITS * r, e_new)
        put_lin(A_NEW_BASE + WORD_BITS * r, a_new)

        st = [a_new, aa, bb, cc, e_new, ee, ff, gg]

    # Output feed-forward: H_out[w] = st[w] + H_in[w], with its own carry row.
    for w in range(H_WORDS):
        out = add(OUT_CARRY_BASE + CARRY_BITS * w, st[w], h_in[:, w])
        put_lin(H_OUT_BASE + WORD_BITS * w, out)

    put(Z_CONST_POS, 1, 1, 1, 1)

    def pack(fields):
        return emit(fields, n, words=WORDS_PER_BLOCK, useful_bits=USEFUL_BITS)

    return pack(zf), pack(af), pack(bf)


def read_words(z_lanes, base: int, count: int):
    """`count` consecutive 32-bit fields starting at bit `base`, out of a packed
    z stream.

    z_lanes: host uint64 [N*256, 2] (the golden loader's lane form) -> uint32
    [N, count]. Every region of this layout that holds 32-bit values is
    word-aligned, so this one reader serves all of them.
    """
    zw = np.asarray(z_lanes, dtype=np.uint64).reshape(-1, WORDS_PER_BLOCK)
    out = np.zeros((zw.shape[0], count), np.uint32)
    for i in range(count):
        w, s = divmod(base + WORD_BITS * i, 64)
        out[:, i] = ((zw[:, w] >> np.uint64(s)) & np.uint64(ONES32)).astype(np.uint32)
    return out


def extract_inputs(z_lanes):
    """Recover every block's compression inputs from a packed z stream.

    z_lanes: host uint64 [N*256, 2] (the golden loader's lane form) ->
    (h_in, m) exactly as `witness_sha2` takes them. Unlike BLAKE3's, this is a
    plain word-aligned read: sha2's constant wire sits at the end of the block
    rather than between the input slots, so nothing is shifted by a bit.
    """
    zw = np.asarray(z_lanes, dtype=np.uint64).reshape(-1, WORDS_PER_BLOCK)
    w, s = divmod(Z_CONST_POS, 64)
    if not np.all((zw[:, w] >> np.uint64(s)) & np.uint64(1)):
        raise ValueError("constant-1 wire missing — not a packed sha2 z stream")
    return read_words(z_lanes, H_BASE, H_WORDS), read_words(z_lanes, M_BASE, M_WORDS)
