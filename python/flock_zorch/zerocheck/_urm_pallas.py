"""Pallas/Triton round-1 URM kernel over the factored row-eq.

The row-eq table of the URM composite factors over the pinned zerocheck
challenges: row = (o << 7) | (j << 3) | k gives eq[row] = EO[o]·MG[j]·SG[k],
with SG geometric under α = φ8(x) and MG geometric under GHASH x. That lets
the small dim fold as a u16 shift-XOR (Σ SG[k]·φ8(y_k) = SG[0]·φ8(Σ y_k≪k)),
the medium dim as a γ-Horner of 1-bit shifts, and leaves ONE GF(2¹²⁸)
multiply per (outer chunk, output column) — so the kernel reads eq_out
(16 B per 128 rows) instead of a materialized eq (16 B per row), and the
128-bit select-XOR accumulate of the composite becomes u16 arithmetic.
Measured on the standalone twin at the 2^22-row geometry: 915 → 389 µs
against the wheel-equivalent form, byte-exact.

The kernel is Triton, so it has no CPU lowering — the caller keeps the
`_round1_partial_decomp` composite as the CPU path and byte oracle, the same
split `witness_blake3` uses. Output keeps `_round1_partials`' contract
(`[n_partials, 2, 64]` ghash over contiguous 512-row ranges), so consumers
and the proof byte gates apply unchanged.

Table forms:
  - t0word  u64[256, 8]: ext(v at byte 0) packed 8 bytes per word,
    word w = ext bytes 8w..8w+7 little-endian. By the circulant
    ext(v at b)[8w+i] == ext(v at 0)[8(w^b)+i], the S→Λ extension of a
    packed row is per-λ word_λ = XOR_b t0word[byte_b(src)][(λ>>3) ^ b]
    followed by ONE byte extract at offset λ&7 — 8 distinct 8-byte
    addresses per gather instead of the 64 distinct 1-byte addresses of
    the per-λ byte-table form (which paid ~8× the L1 transactions).
  - phi_lo/phi_hi u64[256]: φ8 lift lanes.
  - log u8[256] / alog u8[512]: F8 discrete-log multiply (generator 0x03),
    log[0] a sentinel masked by the zero test.
"""

from __future__ import annotations

import functools

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx._src.pallas.triton import primitives as plgpu_prims
from frx.experimental import pallas as pl
from frx.experimental.pallas import triton as plgpu

from flock_zorch import ghash, sumcheck
from flock_zorch.zerocheck import _urm

# Swept at the 2^22-row geometry: 512 rows/program with 2 warps is the
# minimum; 1 warp costs ~21%, 4 warps ~2.7x, and both smaller and larger
# programs lose a few percent.
_ROWS_PER_PARTIAL = 512
_OUTER_PER_PARTIAL = _ROWS_PER_PARTIAL // 128
_NUM_WARPS = 2
_NUM_STAGES = None  # Triton default (3); swept flat

# GF(2^128) GHASH multiply as one inline-PTX block (the twin's byte-validated
# 12-clmad schedule): $0,$1 = out lo/hi; $2,$3 = x lo/hi; $4,$5 = y lo/hi.
_GMUL_ASM = """{
.reg .b64 z, p0, p1, p2, p3;
mov.u64 z, 0;
clmad.lo.u64 p0, $2, $4, z;
clmad.hi.u64 p1, $2, $4, z;
clmad.lo.u64 p1, $3, $4, p1;
clmad.lo.u64 p1, $2, $5, p1;
clmad.lo.u64 p2, $3, $5, z;
clmad.hi.u64 p2, $3, $4, p2;
clmad.hi.u64 p2, $2, $5, p2;
clmad.hi.u64 p3, $3, $5, z;
clmad.lo.u64 p1, p3, 0x87, p1;
clmad.hi.u64 p2, p3, 0x87, p2;
clmad.lo.u64 p0, p2, 0x87, p0;
clmad.hi.u64 p1, p2, 0x87, p1;
mov.u64 $0, p0;
mov.u64 $1, p1;
}"""


def _gmul(x_lo: Array, x_hi: Array, y_lo: Array, y_hi: Array):
    # tt.elementwise_inline_asm has no unsigned tensor type — reinterpret the
    # u64 lanes as i64 across the asm boundary (same bits either way).
    r_lo, r_hi = plgpu_prims.elementwise_inline_asm(
        _GMUL_ASM,
        args=[v.astype(fnp.int64) for v in (x_lo, x_hi, y_lo, y_hi)],
        constraints="=l,=l,l,l,l,l",
        pack=1,
        result_shape_dtypes=[
            frx.ShapeDtypeStruct(x_lo.shape, fnp.int64),
            frx.ShapeDtypeStruct(x_lo.shape, fnp.int64),
        ],
    )
    return r_lo.astype(fnp.uint64), r_hi.astype(fnp.uint64)


def _gf8_reduce(p: Array) -> Array:
    """AES-poly reduce of a <=15-bit value (vector, int32 lanes)."""
    h = p >> 8
    t = (p & 0xFF) ^ h ^ (h << 1) ^ (h << 3) ^ (h << 4)
    h2 = t >> 8
    return (t & 0xFF) ^ h2 ^ (h2 << 1) ^ (h2 << 3) ^ (h2 << 4)


def _xor_reduce0(x: Array) -> Array:
    """XOR-reduce axis 0 (power-of-2 sized) by halving splits.

    The Triton lowering has no reduce_xor; lax.split + elementwise XOR is the
    equivalent log-depth tree."""
    while x.shape[0] > 1:
        half = x.shape[0] // 2
        lo, hi = frx.lax.split(x, (half, half), axis=0)
        x = lo ^ hi
    return x.reshape(x.shape[1:])


def _kernel(
    a_ref,
    b_ref,
    c_ref,
    eo_ref,
    t0_ref,
    plo_ref,
    phi_ref,
    log_ref,
    alog_ref,
    out_ref,
):
    part = pl.program_id(0)
    lam = fnp.arange(64, dtype=fnp.int32)
    w8 = fnp.arange(8, dtype=fnp.int32)  # word index in the t0word circulant
    sh3 = (w8 << 3).astype(fnp.uint64)  # byte offsets within a word
    k8 = w8[:, None]  # small-dim shift-reduce exponents, (8, 1)
    zero64 = fnp.zeros((64,), fnp.uint64)
    acc = [zero64, zero64, zero64, zero64]  # ab_lo, ab_hi, c_lo, c_hi

    del part  # BlockSpec index maps already scope every ref to this program

    def j_body(m, chunk):
        # j runs 15 -> 0 so the gamma-Horner ends as sum_j gamma^j * phi8(.)
        base = chunk[4] * 128 + (15 - m) * 8
        # All 8 small-dim rows at once: the (8, 8) word gathers touch 64
        # distinct addresses each, where a per-row (64,) gather repeats every
        # word across the 8 lanes that share it — 8x the load instructions
        # for the same useful bytes. (Fusing two j-groups per iteration
        # measured flat — the serial Horner loop is not the bound.)
        av = a_ref[pl.ds(base, 8)]
        bv = b_ref[pl.ds(base, 8)]
        cv = c_ref[pl.ds(base, 8)]
        eaw = fnp.zeros((8, 8), fnp.uint64)
        ebw = fnp.zeros((8, 8), fnp.uint64)
        for byte in range(8):
            va = ((av >> (8 * byte)) & 0xFF).astype(fnp.int32)
            vb = ((bv >> (8 * byte)) & 0xFF).astype(fnp.int32)
            idx = w8[None, :] ^ byte
            eaw = eaw ^ t0_ref[va[:, None] * 8 + idx]
            ebw = ebw ^ t0_ref[vb[:, None] * 8 + idx]
        # Byte extract distributes over the word XORs: ea[k, 8w+i] is byte i
        # of eaw[k, w].
        ea = (
            ((eaw[:, :, None] >> sh3[None, None, :]) & fnp.uint64(0xFF))
            .astype(fnp.int32)
            .reshape(8, 64)
        )
        eb = (
            ((ebw[:, :, None] >> sh3[None, None, :]) & fnp.uint64(0xFF))
            .astype(fnp.int32)
            .reshape(8, 64)
        )
        # F8 multiply via discrete log; zero-masked. (A bit-sliced schoolbook
        # form measured ~6% slower — the kernel is ALU-bound and the tiny
        # log/alog tables stay resident in L1.)
        la = log_ref[ea].astype(fnp.int32)
        lb = log_ref[eb].astype(fnp.int32)
        prod = alog_ref[la + lb].astype(fnp.int32)
        nonzero = (ea != 0) & (eb != 0)
        prod = fnp.where(nonzero, prod, 0)
        aacc = _xor_reduce0(prod << k8)
        cbit = ((cv[:, None] >> lam.astype(fnp.uint64)[None, :]) & 1).astype(fnp.int32)
        cacc = _xor_reduce0(cbit << k8)
        sn = _gf8_reduce(aacc)
        out = list(chunk)
        # gamma-Horner: chunk = x*chunk + phi8(sn)
        for t, idx in ((0, sn), (2, cacc)):
            lo, hi = chunk[t], chunk[t + 1]
            carry = hi >> 63
            red = fnp.where(carry != 0, fnp.uint64(0x87), fnp.uint64(0))
            out[t] = ((lo << 1) ^ red) ^ plo_ref[idx]
            out[t + 1] = ((hi << 1) | (lo >> 63)) ^ phi_ref[idx]
        return tuple(out)

    def o_body(o_local, acc):
        chunk = frx.lax.fori_loop(
            0, 16, j_body, (zero64, zero64, zero64, zero64, o_local)
        )
        eo_lo = eo_ref[o_local * 2]
        eo_hi = eo_ref[o_local * 2 + 1]
        eo_lo_v = fnp.full((64,), eo_lo, fnp.uint64)
        eo_hi_v = fnp.full((64,), eo_hi, fnp.uint64)
        ab_lo, ab_hi = _gmul(eo_lo_v, eo_hi_v, chunk[0], chunk[1])
        c_lo, c_hi = _gmul(eo_lo_v, eo_hi_v, chunk[2], chunk[3])
        return (acc[0] ^ ab_lo, acc[1] ^ ab_hi, acc[2] ^ c_lo, acc[3] ^ c_hi)

    acc = frx.lax.fori_loop(0, _OUTER_PER_PARTIAL, o_body, tuple(acc))

    # One output with a trailing (lo, hi) lane axis: `to_ghash` on it is a
    # pure bitcast, where separate lo/hi outputs forced a lane-interleave
    # copy of the whole partials buffer (~268 MB of traffic at m=32).
    out_ref[0, 0, :, 0] = acc[0]
    out_ref[0, 0, :, 1] = acc[1]
    out_ref[0, 1, :, 0] = acc[2]
    out_ref[0, 1, :, 1] = acc[3]


@functools.cache
def _tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """t0word, phi_lo, phi_hi, log, alog — host-built once per process."""
    byte_rows = np.zeros((256, 64), dtype=np.uint8)
    for byte in range(256):
        for col in range(64):
            byte_rows[byte, col] = (byte >> col) & 1 if col < 8 else 0
    # Concrete even under an enclosing trace: the tables are input-independent
    # constants (the @cache would otherwise capture tracers).
    with frx.ensure_compile_time_eval():
        ext = np.asarray(_urm._to_u8(_urm._extend_rows(fnp.asarray(byte_rows), 6)))
    t0byte = ext.astype(np.uint8)  # [256, 64]
    t0word = np.zeros((256, 8), dtype=np.uint64)
    for i in range(8):
        t0word |= t0byte[:, i::8].astype(np.uint64) << np.uint64(8 * i)
    phi = _urm.PHI_8_TABLE  # u64 [256, 2]
    phi_lo = np.ascontiguousarray(phi[:, 0])
    phi_hi = np.ascontiguousarray(phi[:, 1])
    log = np.zeros(256, dtype=np.uint8)
    alog = np.zeros(512, dtype=np.uint8)
    v = 1
    for i in range(255):
        alog[i] = v
        alog[i + 255] = v
        log[v] = i
        v = ((v << 1) ^ ((v >> 7) * 0x1B) ^ v) & 0xFF  # v *= 0x03
    alog[510] = 1
    alog[511] = alog[1]
    return t0word.reshape(-1), phi_lo, phi_hi, log, alog


def round1_partials_pallas(a: Array, b: Array, c: Array, eq_out_scaled: Array) -> Array:
    """Factored-eq URM partials on GPU: `[n_partials, 2, 64]` ghash.

    a/b/c are the packed u64 witness rows (flat, row-major); eq_out_scaled is
    the outer eq table pre-multiplied by SG[0]·MG[0], as u64 lanes
    `[n_out, 2]`. Byte-identical to `_round1_partials` on the same rows by
    the factored-eq identities (asserted in the unit test)."""
    rows = a.shape[0]
    n_partials = rows // _ROWS_PER_PARTIAL
    t0word, phi_lo, phi_hi, log, alog = _tables()
    out = pl.pallas_call(
        _kernel,
        grid=(n_partials,),
        in_specs=[
            pl.BlockSpec((_ROWS_PER_PARTIAL,), lambda p: (p,)),
            pl.BlockSpec((_ROWS_PER_PARTIAL,), lambda p: (p,)),
            pl.BlockSpec((_ROWS_PER_PARTIAL,), lambda p: (p,)),
            pl.BlockSpec((_OUTER_PER_PARTIAL * 2,), lambda p: (p,)),
            pl.BlockSpec((256 * 8,), lambda p: (0,)),
            pl.BlockSpec((256,), lambda p: (0,)),
            pl.BlockSpec((256,), lambda p: (0,)),
            pl.BlockSpec((256,), lambda p: (0,)),
            pl.BlockSpec((512,), lambda p: (0,)),
        ],
        out_specs=pl.BlockSpec((1, 2, 64, 2), lambda p: (p, 0, 0, 0)),
        out_shape=frx.ShapeDtypeStruct((n_partials, 2, 64, 2), fnp.uint64),
        compiler_params=plgpu.CompilerParams(
            num_warps=_NUM_WARPS, num_stages=_NUM_STAGES
        ),
    )(
        a,
        b,
        c,
        eq_out_scaled.reshape(-1),
        fnp.asarray(t0word),
        fnp.asarray(phi_lo),
        fnp.asarray(phi_hi),
        fnp.asarray(log),
        fnp.asarray(alog),
    )
    return ghash.to_ghash(out)


@functools.partial(frx.jit, static_argnums=(3,))
def round1_core_pallas(a, b, c, k_skip, r):
    """`_urm._round1_core` on the factored-eq kernel: (P^AB partial, P^C).

    a/b/c are packed F128 witnesses (uint64 `[2^(m-7), 2]`), whose flat u64
    view is exactly the kernel's row-major 64-bit rows at k_skip=6.

    Precondition (the dispatcher `_urm._round1_pallas_ok` checks it): the
    inner 7 coordinates of `r[k_skip:]` are the protocol's pinned
    small/medium challenges — the kernel's shift-reduce and gamma-Horner
    algebra is specific to those values. The eq factorization itself
    (`eq[row] = EO[o]·MG[j]·SG[k]`) is generic build_eq tensor structure;
    only the geometric-ratio identities need the pin."""
    outer_point = r[k_skip:]
    sg = sumcheck.build_eq(outer_point[:3])
    mg = sumcheck.build_eq(outer_point[3:7])
    eo = sumcheck.build_eq(outer_point[7:])
    eo_scaled = eo * (sg[0] * mg[0])
    partials = round1_partials_pallas(
        a.reshape(-1), b.reshape(-1), c.reshape(-1), ghash.from_ghash(eo_scaled)
    )
    out = fnp.sum(partials, axis=0)
    return out[0], _urm._extend_folded_c(out[1], k_skip)
