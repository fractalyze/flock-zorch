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
  - t0byte  u8[256, 64]: ext(v at byte 0) as bytes; the S→Λ extension of a
    packed row is ext[λ] = XOR_b t0byte[byte_b(src)][λ ^ (b<<3)] — the
    word-level circulant ext(v at b)[8w+i] == ext(v at 0)[8(w^b)+i]
    restated per byte, a pure per-λ gather.
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

from flock_zorch import ghash
from flock_zorch.zerocheck import _urm

_ROWS_PER_PARTIAL = 128  # one outer chunk per program: max grid parallelism
_OUTER_PER_PARTIAL = _ROWS_PER_PARTIAL // 128

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
    out_lo_ref,
    out_hi_ref,
):
    part = pl.program_id(0)
    lam = fnp.arange(64, dtype=fnp.int32)
    zero64 = fnp.zeros((64,), fnp.uint64)
    acc = [zero64, zero64, zero64, zero64]  # ab_lo, ab_hi, c_lo, c_hi

    del part  # BlockSpec index maps already scope every ref to this program

    def j_body(m, chunk):
        # j runs 15 -> 0 so the gamma-Horner ends as sum_j gamma^j * phi8(.)
        base = chunk[4] * 128 + (15 - m) * 8
        aacc = fnp.zeros((64,), fnp.int32)
        cacc = fnp.zeros((64,), fnp.int32)
        for k in range(8):
            src_a = a_ref[base + k]
            src_b = b_ref[base + k]
            src_c = c_ref[base + k]
            ea = fnp.zeros((64,), fnp.int32)
            eb = fnp.zeros((64,), fnp.int32)
            for byte in range(8):
                va = (src_a >> (8 * byte)).astype(fnp.int32) & 0xFF
                vb = (src_b >> (8 * byte)).astype(fnp.int32) & 0xFF
                col = lam ^ (byte << 3)
                ea = ea ^ t0_ref[va * 64 + col].astype(fnp.int32)
                eb = eb ^ t0_ref[vb * 64 + col].astype(fnp.int32)
            # F8 multiply via discrete log; zero-masked.
            la = log_ref[ea].astype(fnp.int32)
            lb = log_ref[eb].astype(fnp.int32)
            prod = alog_ref[la + lb].astype(fnp.int32)
            nonzero = (ea != 0) & (eb != 0)
            prod = fnp.where(nonzero, prod, 0)
            aacc = aacc ^ (prod << k)
            cbit = ((src_c >> lam.astype(fnp.uint64)) & 1).astype(fnp.int32)
            cacc = cacc ^ (cbit << k)
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

    out_lo_ref[0, 0, :] = acc[0]
    out_hi_ref[0, 0, :] = acc[1]
    out_lo_ref[0, 1, :] = acc[2]
    out_hi_ref[0, 1, :] = acc[3]


@functools.cache
def _tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """t0byte, phi_lo, phi_hi, log, alog — host-built once per process."""
    byte_rows = np.zeros((256, 64), dtype=np.uint8)
    for byte in range(256):
        for col in range(64):
            byte_rows[byte, col] = (byte >> col) & 1 if col < 8 else 0
    # Concrete even under an enclosing trace: the tables are input-independent
    # constants (the @cache would otherwise capture tracers).
    with frx.ensure_compile_time_eval():
        ext = np.asarray(_urm._to_u8(_urm._extend_rows(fnp.asarray(byte_rows), 6)))
    t0byte = ext.astype(np.uint8)  # [256, 64]
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
    return t0byte.reshape(-1), phi_lo, phi_hi, log, alog


def round1_partials_pallas(a: Array, b: Array, c: Array, eq_out_scaled: Array) -> Array:
    """Factored-eq URM partials on GPU: `[n_partials, 2, 64]` ghash.

    a/b/c are the packed u64 witness rows (flat, row-major); eq_out_scaled is
    the outer eq table pre-multiplied by SG[0]·MG[0], as u64 lanes
    `[n_out, 2]`. Byte-identical to `_round1_partials` on the same rows by
    the factored-eq identities (asserted in the unit test)."""
    rows = a.shape[0]
    n_partials = rows // _ROWS_PER_PARTIAL
    t0byte, phi_lo, phi_hi, log, alog = _tables()
    out_lo, out_hi = pl.pallas_call(
        _kernel,
        grid=(n_partials,),
        in_specs=[
            pl.BlockSpec((_ROWS_PER_PARTIAL,), lambda p: (p,)),
            pl.BlockSpec((_ROWS_PER_PARTIAL,), lambda p: (p,)),
            pl.BlockSpec((_ROWS_PER_PARTIAL,), lambda p: (p,)),
            pl.BlockSpec((_OUTER_PER_PARTIAL * 2,), lambda p: (p,)),
            pl.BlockSpec((256 * 64,), lambda p: (0,)),
            pl.BlockSpec((256,), lambda p: (0,)),
            pl.BlockSpec((256,), lambda p: (0,)),
            pl.BlockSpec((256,), lambda p: (0,)),
            pl.BlockSpec((512,), lambda p: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((1, 2, 64), lambda p: (p, 0, 0)),
            pl.BlockSpec((1, 2, 64), lambda p: (p, 0, 0)),
        ],
        out_shape=[
            frx.ShapeDtypeStruct((n_partials, 2, 64), fnp.uint64),
            frx.ShapeDtypeStruct((n_partials, 2, 64), fnp.uint64),
        ],
        compiler_params=plgpu.CompilerParams(num_warps=1),
    )(
        a,
        b,
        c,
        eq_out_scaled.reshape(-1),
        fnp.asarray(t0byte),
        fnp.asarray(phi_lo),
        fnp.asarray(phi_hi),
        fnp.asarray(log),
        fnp.asarray(alog),
    )
    lanes = fnp.stack([out_lo, out_hi], axis=-1)  # [n_partials, 2, 64, 2]
    return ghash.to_ghash(lanes)
