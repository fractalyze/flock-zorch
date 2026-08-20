"""GF(2¹²⁸) GHASH multiply for Pallas/Triton kernels — the one inline-PTX block.

Pallas/Triton has no binary-field dtype, so a kernel that needs a field
multiply must spell it. This is the 12-`clmad` schedule (schoolbook u64
quarters + the 0x87 reduction folded into the same instruction), byte-validated
against the native `binary_field_ghash` multiply by the URM twin harness and
transitively by every proof gate that runs a kernel using it.

This module is deliberately the ONLY hardware-specific dependency of the
in-repo kernel set: `clmad` is PTX ISA 9.3+ / sm_120+, and Triton assembles
with its own bundled ptxas (NOT the one XLA resolves — `docs/measurement.md`'s
toolchain rules govern XLA only), so on a too-old wheel or card a kernel using
this fails at compile rather than degrading. Callers keep a portable
marker/composite path as the fallback and byte oracle; keep it that way.

A software shift/XOR multiply would remove the requirement but not earn its
keep: the clmad-form reduce alone measured −35% against the production
multiply on the URM twin, and a full schoolbook is far worse.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
from frx import Array
from frx._src.pallas.triton import primitives as plgpu_prims

# $0,$1 = out lo/hi; $2,$3 = x lo/hi; $4,$5 = y lo/hi.
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


def gmul(x_lo: Array, x_hi: Array, y_lo: Array, y_hi: Array):
    """Lanewise GHASH multiply of uint64 (lo, hi) pairs, inside a kernel body."""
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
