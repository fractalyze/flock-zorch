"""Device (GPU) round-1 URM core — the F8 S→Λ extension via the compiler's
field-generic additive NTT (`lax.ntt` over `binary_field_gf8_aes`), plus the
φ8 lift and F128 eq-accumulation, fused in one jit kernel. Byte-identical to
flock's `round1_naive` (gated by the URM oracle); the heavy [N,ell,2] φ8
intermediate is consumed in-fusion (never written to HBM).

The extension uses base-subspace transforms only: inverse NTT size ℓ →
zero-pad coefficients to 2ℓ → forward NTT size 2ℓ → second half = the β=ℓ
coset. No coset-offset kernel support needed.

Reads _urm's host PHI_8_TABLE (the only host<->device shared datum). _urm ⇄ _urm_device
is a deliberate cycle made safe by MODULE imports on both sides (no import-time lookup
of a not-yet-defined name) + call-time attribute access, so either load order works.

Requires jax_enable_x64.
"""
from __future__ import annotations

import functools

import numpy as np
import frx
import frx.numpy as jnp
import zk_dtypes
from frx import lax

from flock_zorch import field
from flock_zorch.zerocheck import _urm

_PHI_DEV = jnp.asarray(_urm.PHI_8_TABLE)     # [256, 2] uint64
_PHI_DEV_G = field.to_ghash(_PHI_DEV)        # [256] ghash — indexed in-kernel, no lane bitcast
_AES = np.dtype(zk_dtypes.binary_field_gf8_aes)


def _extend_rows(rows, k_skip: int):
    """S→Λ extension, uint8 rows [N, 2^k_skip] -> AES-dtype rows on Λ."""
    ell = 1 << k_skip
    v = lax.bitcast_convert_type(rows, _AES)
    coeffs = lax.ntt(v, ntt_type="INTT", ntt_length=ell)
    padded = jnp.concatenate([coeffs, jnp.zeros_like(coeffs)], axis=-1)
    evals = lax.ntt(padded, ntt_type="NTT", ntt_length=2 * ell)
    return evals[..., ell:]


def _to_u8(x):
    return lax.bitcast_convert_type(x, jnp.uint8)


_R1_CORE = None


def _round1_core():
    """Fused round-1 core (memoized): extend a/b/c S→Λ, a·b, φ8-embed, AND
    eq-accumulate — all in ONE jit kernel so the large [N,ell,2] φ8 intermediate is
    consumed in-fusion and never written to HBM (halves round1's bandwidth vs the
    separate extend + accumulate)."""
    global _R1_CORE
    if _R1_CORE is None:
        @functools.partial(frx.jit, static_argnums=(3,))
        def core(a, b, c, k_skip, eqx):
            a_l = _extend_rows(a, k_skip)
            b_l = _extend_rows(b, k_skip)
            c_l = _to_u8(_extend_rows(c, k_skip))
            ab = _to_u8(a_l * b_l).astype(jnp.int32)
            phi_ab = _PHI_DEV_G[ab]
            phi_c = _PHI_DEV_G[c_l.astype(jnp.int32)]
            return (field.from_ghash(jnp.sum(eqx * phi_ab, axis=0)),
                    field.from_ghash(jnp.sum(eqx * phi_c, axis=0)))
        _R1_CORE = core
    return _R1_CORE


@functools.partial(frx.jit, static_argnums=(1, 2))
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
