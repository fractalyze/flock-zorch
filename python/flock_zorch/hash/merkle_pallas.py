"""Pallas/Triton batched SHA-256 leaf hash over raw leaf bytes.

`_Sha256MerkleTree._hash_leaves` pays a full-buffer pad-and-pack pass before
the `hash_frx.sha256` marker: `_pad_device` concatenates the padding suffix
and assembles big-endian u32 words, materializing `[B, nblocks, 16]` u32 —
~1.16 GB written and re-read at the m=32 l0 commit shape, the single largest
commit-window kernel (2.16 ms) after the NTT itself. This kernel reads the
raw leaf bytes and does the byte-swap in-register; the padding block is a
compile-time constant (static leaf length, multiple of 64), so it is never
read or written at all.

GPU only (Triton has no CPU lowering); the marker path stays the portable
form and the byte oracle, the same split `witness_blake3` and the URM pallas
kernel use. Byte-identical to FIPS 180-4 per leaf — asserted against
`hash_frx`'s digest in the unit test and transitively by the proof gates
(one moved byte changes the Merkle root and every draw after it).
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.experimental import pallas as pl
from frx.experimental.pallas import triton as plgpu

_LEAVES_PER_PROG = 64
_NUM_WARPS = 2

# FIPS 180-4 round constants and initial state (host constants; K is baked
# into the unrolled round bodies as immediates).
_K = [
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
]
_H0 = [
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
]


def _rotr(x: Array, n: int) -> Array:
    return (x >> n) | (x << (32 - n))


def _bswap32(x: Array) -> Array:
    m = fnp.uint32(0xFF)
    return ((x >> 24) & m) | ((x >> 8) & (m << 8)) | ((x << 8) & (m << 16)) | (x << 24)


def _compress(state: tuple, w16: list) -> tuple:
    """One 64-round SHA-256 compression over a 16-word block (vector lanes)."""
    a, b, c, d, e, f, g, h = state
    w = list(w16)
    for r in range(64):
        if r >= 16:
            s0 = (
                _rotr(w[(r - 15) % 16], 7)
                ^ _rotr(w[(r - 15) % 16], 18)
                ^ (w[(r - 15) % 16] >> 3)
            )
            s1 = (
                _rotr(w[(r - 2) % 16], 17)
                ^ _rotr(w[(r - 2) % 16], 19)
                ^ (w[(r - 2) % 16] >> 10)
            )
            w[r % 16] = w[r % 16] + s0 + w[(r - 7) % 16] + s1
        e1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = h + e1 + ch + fnp.uint32(_K[r]) + w[r % 16]
        a0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = a0 + maj
        a, b, c, d, e, f, g, h = t1 + t2, a, b, c, d + t1, e, f, g
    # Davies-Meyer feed-forward: the block's output chains as state + rounds.
    return tuple(s + v for s, v in zip(state, (a, b, c, d, e, f, g, h)))


def _make_kernel(length: int):
    """Kernel for static leaf byte-length `length` (a multiple of 64)."""
    n_data_blocks = length // 64
    # The constant padding block: 0x80, zeros, 8-byte BE bit length.
    tail = np.zeros(64, dtype=np.uint8)
    tail[0] = 0x80
    tail[-8:] = np.frombuffer((length * 8).to_bytes(8, "big"), np.uint8)
    pad_words = [int(x) for x in tail.view(">u4")]

    def kernel(in_ref, out_ref):
        lanes = fnp.arange(_LEAVES_PER_PROG, dtype=fnp.int32)
        state = tuple(fnp.full((_LEAVES_PER_PROG,), h, fnp.uint32) for h in _H0)
        words_per_leaf = length // 4

        def block_body(blk, state):
            # One (leaves, 16) tile: rows are 64-byte runs, decently coalesced.
            idx = (
                lanes[:, None] * words_per_leaf
                + blk * 16
                + fnp.arange(16, dtype=fnp.int32)
            )
            tile = _bswap32(in_ref[idx])  # LE storage -> BE message words
            w16 = [
                col.reshape(_LEAVES_PER_PROG)
                for col in frx.lax.split(tile, (1,) * 16, axis=1)
            ]
            return _compress(state, w16)

        state = frx.lax.fori_loop(0, n_data_blocks, block_body, state)
        w16 = [fnp.full((_LEAVES_PER_PROG,), w, fnp.uint32) for w in pad_words]
        final = _compress(state, w16)
        for i in range(8):
            out_ref[:, i] = _bswap32(final[i])  # BE digest serialization

    return kernel


def sha256_leaves_pallas(rows: Array) -> Array:
    """Batched SHA-256 of equal-length rows: uint8 `[B, L]` -> uint8 `[B, 32]`.

    Requires L % 64 == 0 and B % 64 == 0 (whole tiles / programs — the
    dispatcher guards). Byte-identical to `hash_frx.sha256.digest`."""
    b, length = rows.shape
    assert length % 64 == 0 and b % _LEAVES_PER_PROG == 0
    words = frx.lax.bitcast_convert_type(
        rows.reshape(b, length // 4, 4), fnp.uint32
    ).reshape(-1)
    out = pl.pallas_call(
        _make_kernel(length),
        grid=(b // _LEAVES_PER_PROG,),
        in_specs=[pl.BlockSpec((_LEAVES_PER_PROG * (length // 4),), lambda p: (p,))],
        out_specs=pl.BlockSpec((_LEAVES_PER_PROG, 8), lambda p: (p, 0)),
        out_shape=frx.ShapeDtypeStruct((b, 8), fnp.uint32),
        compiler_params=plgpu.CompilerParams(num_warps=_NUM_WARPS),
    )(words)
    return frx.lax.bitcast_convert_type(out, fnp.uint8).reshape(b, 32)
