"""flock GF(2^128) arithmetic in the GHASH basis, authored in jax. Ports flock-core
`field/gf2_128.rs` (the `software` module: `clmul64` / `ghash_mul_unreduced` /
`ghash_mul`, and the free `ghash_reduce`); see the flock paper §4.3 (field).

flock (succinctlabs/flock) represents F128 = GF(2^128) in the GHASH/POLYVAL
polynomial basis: irreducible p(x) = x^128 + x^7 + x^2 + x + 1 (reduction
constant 0x87), natural (non-bit-reflected) bit order, adjoined root γ = x
(so F128(lo=2) = x). An element is two u64 limbs {lo, hi} where `lo` holds the
coefficients of x^0..x^63 and `hi` of x^64..x^127.

zk_dtypes ships `binary_field_t7` (GF(2^128)) but in the x^2+x+a TOWER basis,
a DIFFERENT (isomorphic, not bit-compatible) representation -- e.g. 2*2 = 3
there vs 4 in flock. flock hashes raw field bytes pervasively (Merkle leaves,
its SHA-256 transcript), so byte-identity with the flock verifier requires
arithmetic in flock's basis. We therefore implement GHASH multiply directly
over uint64 lanes; the result matches flock bit-for-bit.

Representation: an F128 array is uint64 of shape [..., 2] = [lo, hi]. On a
little-endian host `np.asarray(x).tobytes()` equals flock's
`lo.to_le_bytes() ++ hi.to_le_bytes()` 16-byte serialization.

The carryless product is bit-serial via `lax.fori_loop` (NOT a Python unroll):
the 64 steps stay one while-loop body, so the kernel is O(1) memory and scales
to flock's production codeword sizes (2^23+). A fully-unrolled version OOMs --
XLA materializes the whole 256-step chain. This software multiply is correct but
not the fast path: GPU acceleration is the PTX `clmad` (carryless multiply-ADD;
PTX has no standalone CLMUL, only the fused multiply-add -- confirmed absent from
the PTX ISA + the in-tree LLVM), wired in the zkx compiler rather than here --
perf is the compiler's job (benchmarks in optim/clmad/README.md).

Requires `jax_enable_x64`.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

U64 = jnp.uint64
_ONE = U64(1)

LOG_PACKING = 7  # an F128 packs 2^7 = 128 bits; witness log-size m -> 2^(m-7) packed elems


def add(a, b):
    """GF(2^128) addition is bitwise XOR. a, b: uint64 [..., 2]."""
    return a ^ b


def sum(x, axis: int = 0):
    """GF(2^128) summation over an axis: add is XOR, so Σ is an XOR-reduce.
    Lowers to one XLA reduce (log-depth, O(1) memory)."""
    return jax.lax.reduce(x, U64(0), jax.lax.bitwise_xor, (axis,))


def _clmul64(a, b):
    """64x64 -> 128 carryless (GF(2)[x]) product. a, b: uint64 [...] -> (lo, hi).

    Bit-serial accumulation as a single while-loop (`lax.fori_loop`), so it lowers
    to one O(1)-memory kernel that scales to large batches; a Python `for` unroll
    OOMs at 2^23. For step i with bit i of `a` set, XOR in `b << i` (low limb) and
    `b >> (64-i)` (high limb); i==0 contributes nothing to the high limb.
    """
    z = jnp.zeros_like(a)

    def body(i, carry):
        lo, hi = carry
        iu = i.astype(U64)
        mask = U64(0) - ((a >> iu) & _ONE)  # all-ones iff bit i of a is set
        lo = lo ^ (mask & (b << iu))
        sh = jnp.where(i != 0, U64(64) - iu, U64(0))  # avoid the i==0 shift-by-64 UB
        hi = hi ^ jnp.where(i != 0, mask & (b >> sh), z)
        return (lo, hi)

    return jax.lax.fori_loop(0, 64, body, (z, z))


def _mul_unreduced(alo, ahi, blo, bhi):
    """Schoolbook 128x128 -> 256 carryless product (flock ghash_mul_unreduced)."""
    ll_lo, ll_hi = _clmul64(alo, blo)
    lh_lo, lh_hi = _clmul64(alo, bhi)
    hl_lo, hl_hi = _clmul64(ahi, blo)
    hh_lo, hh_hi = _clmul64(ahi, bhi)
    cr_lo = lh_lo ^ hl_lo
    cr_hi = lh_hi ^ hl_hi
    r0 = ll_lo
    r1 = ll_hi ^ cr_lo
    r2 = hh_lo ^ cr_hi
    r3 = hh_hi
    return r0, r1, r2, r3


def _ghash_reduce(r0, r1, r2, r3):
    """Reduce 256 bits mod x^128 + x^7 + x^2 + x + 1 (flock ghash_reduce).

    Fold the high half (r2:r3) down via x^128 == x^7 + x^2 + x + 1, with a 7-bit
    overflow correction for bits pushed past x^127.
    """
    s1_lo = r2 << U64(1)
    s1_hi = (r3 << U64(1)) | (r2 >> U64(63))
    s2_lo = r2 << U64(2)
    s2_hi = (r3 << U64(2)) | (r2 >> U64(62))
    s7_lo = r2 << U64(7)
    s7_hi = (r3 << U64(7)) | (r2 >> U64(57))
    t_lo = r2 ^ s1_lo ^ s2_lo ^ s7_lo
    t_hi = r3 ^ s1_hi ^ s2_hi ^ s7_hi
    ov = (r3 >> U64(63)) ^ (r3 >> U64(62)) ^ (r3 >> U64(57))
    corr = ov ^ (ov << U64(1)) ^ (ov << U64(2)) ^ (ov << U64(7))
    return r0 ^ t_lo ^ corr, r1 ^ t_hi


def mul(a, b):
    """Elementwise GF(2^128) multiply in flock's GHASH basis.

    a, b: uint64 [..., 2] (lo, hi), broadcastable on the leading dims (e.g. a
    scalar [2] against a [N, 2] table). Returns the broadcast shape. Both operands
    are broadcast to a common shape first so the bit-serial `_clmul64` carry stays
    consistent regardless of operand order.
    """
    shape = jnp.broadcast_shapes(a.shape, b.shape)
    a = jnp.broadcast_to(a, shape)
    b = jnp.broadcast_to(b, shape)
    alo, ahi = a[..., 0], a[..., 1]
    blo, bhi = b[..., 0], b[..., 1]
    r0, r1, r2, r3 = _mul_unreduced(alo, ahi, blo, bhi)
    lo, hi = _ghash_reduce(r0, r1, r2, r3)
    return jnp.stack([lo, hi], axis=-1)


# ---- host int <-> uint64-lane serialization (bit i = coefficient of x^i) ----
_MASK64 = (1 << 64) - 1


def _to_int(arr) -> int:
    """F128 uint64 [.., 2] (lo, hi) -> Python int (bit i = coefficient of x^i)."""
    a = np.asarray(arr, dtype=np.uint64)
    return int(a[0]) | (int(a[1]) << 64)


def _to_lohi(x: int) -> np.ndarray:
    """Python-int F128 -> uint64 [2] (lo, hi)."""
    return np.array([x & _MASK64, (x >> 64) & _MASK64], dtype=np.uint64)
