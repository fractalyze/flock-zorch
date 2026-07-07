"""Device (GPU) F8 kernels for the zerocheck round-1 URM — the same φ8 table +
additive NTT as gf8's host reference, in jnp, so round1 runs on the GPU instead of
host numpy. Byte-identical to gf8's host path (gated by the URM oracle); the heavy
[N,ell,2] φ8 intermediate is consumed in-fusion (never written to HBM).

Reads gf8's host PHI_8_TABLE (the only host<->device shared datum). gf8 ⇄ _gf8_device
is a deliberate cycle made safe by MODULE imports on both sides (no import-time lookup
of a not-yet-defined name) + call-time attribute access, so either load order works.

Requires jax_enable_x64.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from flock_zorch import field, gf8

_PHI_DEV = jnp.asarray(gf8.PHI_8_TABLE)     # [256, 2] uint64


def _gf8_mul_dev(a, b):
    """Elementwise F8 multiply on device — ARITHMETIC (clmul8 + mod-0x11B reduce),
    no table gather. A 256x256 table gather over the F8-NTT's elements is
    memory-bound; 8 unrolled XOR-shifts + two reduction folds is compute-bound and
    faster. Byte-identical to the `_MUL` table (same `_clmul8`/`_gf8_reduce` math)."""
    a16 = a.astype(jnp.uint16)
    b16 = b.astype(jnp.uint16)
    p = jnp.zeros_like(a16)
    for i in range(8):
        p = p ^ jnp.where(((a16 >> i) & 1) != 0, b16 << i, jnp.uint16(0))
    h = p >> 8
    t = (p & 0xFF) ^ h ^ (h << 1) ^ (h << 3) ^ (h << 4)
    h2 = t >> 8
    return (((t & 0xFF) ^ h2 ^ (h2 << 1) ^ (h2 << 3) ^ (h2 << 4)) & 0xFF).astype(jnp.uint8)


def _fft_dev(v, tw, k: int):
    """Iterative DIF additive-NTT over F8 (device); v: uint8 [N, 2^k]. Equivalent
    to the recursive `_fft`: level L butterflies 2^L blocks with binary-heap
    twiddles tw[2^L-1 : 2^(L+1)-1]."""
    n, ell = v.shape[0], 1 << k
    for level in range(k):
        nn, block = 1 << level, ell >> level
        half = block // 2
        lam = tw[(1 << level) - 1:(1 << (level + 1)) - 1]      # [nn]
        vr = v.reshape(n, nn, 2, half)
        lo, hi = vr[:, :, 0, :], vr[:, :, 1, :]                # [n, nn, half]
        lam_b = jnp.broadcast_to(lam[None, :, None], lo.shape)
        new_lo = lo ^ _gf8_mul_dev(lam_b, hi)
        new_hi = hi ^ new_lo
        v = jnp.stack([new_lo, new_hi], axis=2).reshape(n, ell)
    return v


def _ifft_dev(v, tw, k: int):
    """Iterative DIT inverse additive-NTT over F8 (device); deepest level first."""
    n, ell = v.shape[0], 1 << k
    for level in reversed(range(k)):
        nn, block = 1 << level, ell >> level
        half = block // 2
        lam = tw[(1 << level) - 1:(1 << (level + 1)) - 1]
        vr = v.reshape(n, nn, 2, half)
        lo, hi = vr[:, :, 0, :], vr[:, :, 1, :]
        lam_b = jnp.broadcast_to(lam[None, :, None], hi.shape)
        new_hi = hi ^ lo
        new_lo = lo ^ _gf8_mul_dev(lam_b, new_hi)
        v = jnp.stack([new_lo, new_hi], axis=2).reshape(n, ell)
    return v


_R1_CORE = None


def _round1_core():
    """Fused round-1 core (memoized): extend a/b/c S→Λ, a·b, φ8-embed, AND
    eq-accumulate — all in ONE jit kernel so the large [N,ell,2] φ8 intermediate is
    consumed in-fusion and never written to HBM (halves round1's bandwidth vs the
    separate extend + accumulate)."""
    global _R1_CORE
    if _R1_CORE is None:
        @functools.partial(jax.jit, static_argnums=(3,))
        def core(a, b, c, k_skip, tw_s, tw_l, eqx):
            a_l = _fft_dev(_ifft_dev(a, tw_s, k_skip), tw_l, k_skip)
            b_l = _fft_dev(_ifft_dev(b, tw_s, k_skip), tw_l, k_skip)
            c_l = _fft_dev(_ifft_dev(c, tw_s, k_skip), tw_l, k_skip)
            phi_ab = field.to_ghash(_PHI_DEV[_gf8_mul_dev(a_l, b_l).astype(jnp.int32)])
            phi_c = field.to_ghash(_PHI_DEV[c_l.astype(jnp.int32)])
            eqx_g = field.to_ghash(eqx)                        # [n_chunks, 1]
            return (field.from_ghash(jnp.sum(eqx_g * phi_ab, axis=0)),
                    field.from_ghash(jnp.sum(eqx_g * phi_c, axis=0)))
        _R1_CORE = core
    return _R1_CORE


@functools.partial(jax.jit, static_argnums=(1, 2))
def _packed_to_rows(packed, m: int, k_skip: int):
    """Packed F128 witness [2^(m-7), 2] uint64 -> uint8 rows [2^(m-k_skip), 2^k_skip],
    unpacked ON DEVICE (bit r of element i = z[i·128 + r], LSB-first per lane).

    The witness is 1/8 the size packed (one F128 lane vs one byte per bit), so
    taking the packed form and unpacking here turns a fat host->device transfer
    into a small one + a cheap device kernel — the same device-unpack pattern
    `prover._unpack_bits_dev` uses for the identity path."""
    bi = jnp.arange(64, dtype=jnp.uint64)
    lo = ((packed[:, 0:1] >> bi) & jnp.uint64(1)).astype(jnp.uint8)
    hi = ((packed[:, 1:2] >> bi) & jnp.uint64(1)).astype(jnp.uint8)
    bits = jnp.concatenate([lo, hi], axis=1).reshape(-1)        # [2^m]
    return bits.reshape(1 << (m - k_skip), 1 << k_skip)
