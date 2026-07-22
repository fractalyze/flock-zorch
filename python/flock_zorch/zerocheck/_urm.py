"""φ₈ embedding of F8 = GF(2⁸) (AES field) into F128 + the zerocheck round-1
univariate-skip URM (`round1_naive`) — host orchestration and the fused device
core. Byte-identical to flock-core's `field/phi8.rs` and
`zerocheck/univariate_skip.rs::round1_naive`.

F8 arithmetic and its additive NTT are compiler-native: the
`binary_field_gf8_aes` dtype (AES poly x⁸+x⁴+x³+x+1 = 0x11B) dispatches the
field-generic LCH14 additive NTT through `lax.ntt`, so this module carries no
field code — only φ₈ (a field homomorphism into an F128 subfield, the only
link between the AES basis and the GHASH basis) and the round-1 plumbing.

The S→Λ extension is `INTT ℓ → coset-NTT ℓ` at β=ℓ: the inverse NTT recovers the
degree-<ℓ coefficients, and the forward coset NTT evaluates them directly on the
Λ = β+S coset (β = ℓ) via lax.ntt's `coset=` (fractalyze/xla #307) — replacing the
old zero-pad-to-2ℓ + size-2ℓ NTT + discard-half trick.

Requires jax_enable_x64.
"""
from __future__ import annotations

import functools

import numpy as np
import frx
import frx.numpy as fnp
import zk_dtypes
from frx import lax

from flock_zorch import ghash, sumcheck

# ---------------------------------------------------------------------------
# phi8: F8 -> F128 embedding (256-entry table). F2-linear, so the full table is
# built by XOR over set bits from the 8 basis images phi8(2^t). Cross-checked
# against flock's PHI_8_TABLE in the URM oracle gate.
# ---------------------------------------------------------------------------

_PHI8_BASIS = np.array([
    [0x0000000000000001, 0x0000000000000000],  # phi8(0x01)
    [0x6B8330483C2E9849, 0x0DCB364640A222FE],  # phi8(0x02)
    [0x7573DA4A5F7710ED, 0x3D5BD35C94646A24],  # phi8(0x04)
    [0x41A12DB1F974F3AC, 0x6D58C4E181F9199F],  # phi8(0x08)
    [0x5E2F716F4EDE412F, 0xA72EC17764D7CED5],  # phi8(0x10)
    [0x5CB10FBABCF00118, 0x4D52354A3A3D8C86],  # phi8(0x20)
    [0x95ED1F57F3632D4D, 0x553E92E8BC0AE9A7],  # phi8(0x40)
    [0x512625B1F09FA87E, 0x93252331BF042B11],  # phi8(0x80)
], dtype=np.uint64)


def _build_phi8_table() -> np.ndarray:
    table = np.zeros((256, 2), dtype=np.uint64)
    for v in range(256):
        acc = np.zeros(2, dtype=np.uint64)
        for t in range(8):
            if (v >> t) & 1:
                acc ^= _PHI8_BASIS[t]
        table[v] = acc
    return table


PHI_8_TABLE = _build_phi8_table()  # uint64 [256, 2] = F128 (host; `_fold` indexes it)

_PHI_DEV = fnp.asarray(PHI_8_TABLE)
_PHI_DEV_G = ghash.to_ghash(_PHI_DEV)        # [256] ghash — indexed in-kernel, no lane bitcast
_AES = np.dtype(zk_dtypes.binary_field_gf8_aes)


# ---------------------------------------------------------------------------
# Device (GPU) round-1 URM core.
# ---------------------------------------------------------------------------


def _extend_rows(rows, k_skip: int):
    """S→Λ extension, uint8 rows [N, 2^k_skip] -> AES-dtype rows on Λ."""
    ell = 1 << k_skip
    v = lax.bitcast_convert_type(rows, _AES)
    coeffs = lax.ntt(v, ntt_type="INTT", ntt_length=ell)
    return lax.ntt(coeffs, ntt_type="NTT", ntt_length=ell, coset=ell)


def _to_u8(x):
    return lax.bitcast_convert_type(x, fnp.uint8)


@functools.partial(frx.jit, static_argnums=(3,))
def _round1_core(a, b, c, k_skip, r):
    """Fused round-1 core: build eqx, extend a/b/c S→Λ, a·b, φ8-embed, AND
    eq-accumulate — all in ONE jit kernel so the large [N,ell,2] φ8 intermediate is
    consumed in-fusion and never written to HBM (halves round1's bandwidth vs the
    separate extend + accumulate). `build_eq` is in-kernel (no `build_eq_fused`)."""
    eqx = sumcheck.build_eq(r[k_skip:])[:, None]           # r is ghash [m]; [n_chunks, 1]
    a_l = _extend_rows(a, k_skip)
    b_l = _extend_rows(b, k_skip)
    c_l = _to_u8(_extend_rows(c, k_skip))
    ab = _to_u8(a_l * b_l).astype(fnp.int32)
    phi_ab = _PHI_DEV_G[ab]
    phi_c = _PHI_DEV_G[c_l.astype(fnp.int32)]
    return (ghash.from_ghash(fnp.sum(eqx * phi_ab, axis=0)),
            ghash.from_ghash(fnp.sum(eqx * phi_c, axis=0)))


@functools.partial(frx.jit, static_argnums=(1, 2))
def _packed_to_rows(packed, m: int, k_skip: int):
    """Packed F128 witness [2^(m-7), 2] uint64 -> uint8 rows [2^(m-k_skip), 2^k_skip],
    unpacked ON DEVICE (bit r of element i = z[i·128 + r], LSB-first per lane).

    The witness is 1/8 the size packed (one F128 lane vs one byte per bit), so
    taking the packed form and unpacking here turns a fat host->device transfer
    into a small one + a cheap device kernel — the same device-unpack pattern
    `prover._unpack_bits` uses for the identity path."""
    bi = fnp.arange(64, dtype=fnp.uint64)
    lo = ((packed[:, 0:1] >> bi) & fnp.uint64(1)).astype(fnp.uint8)
    hi = ((packed[:, 1:2] >> bi) & fnp.uint64(1)).astype(fnp.uint8)
    bits = fnp.concatenate([lo, hi], axis=1).reshape(-1)        # [2^m]
    return bits.reshape(1 << (m - k_skip), 1 << k_skip)


# ---------------------------------------------------------------------------
# round1_naive — the zerocheck round-1 URM reference (== the wire round1_ab/c).
# ---------------------------------------------------------------------------


def witness_to_rows(bits, m: int, k_skip: int):
    """Witness -> device uint8 rows [2^(m-k_skip), 2^k_skip], for round1 + fold_at_z.

    Accepts three forms: the **packed F128** witness (uint64 [2^(m-7), 2]) — unpacked
    on device (8x less host transfer, the preferred form); a uint8 [2^m] (0/1) bit
    array (transferred once); or an already-device array (reshaped, no copy)."""
    n_chunks, ell = 1 << (m - k_skip), 1 << k_skip
    if getattr(bits, "ndim", 0) == 2 and bits.shape[-1] == 2 and np.dtype(bits.dtype) == np.uint64:
        return _packed_to_rows(fnp.asarray(bits), m, k_skip)   # packed F128 -> device unpack
    if isinstance(bits, frx.Array):
        return bits.reshape(n_chunks, ell)
    return fnp.asarray(np.asarray(bits, np.uint8).reshape(n_chunks, ell))


def round1_rows(a, b, c, m: int, k_skip: int, r):
    """Round-1 URM from device witness rows (uint8 [2^(m-k_skip), 2^k_skip]). The
    compute half of `round1_naive`, so the witness can be transferred once and
    reused by `zerocheck._fold_at_z`. Returns (P^AB, P^C) as numpy."""
    p_ab, p_c = _round1_core(a, b, c, k_skip, r)  # eqx build + extend+phi+accum, fused
    return np.asarray(p_ab), np.asarray(p_c)


def round1_naive(a_bits, b_bits, c_bits, m: int, k_skip: int, r):
    """Round-1 univariate-skip message (P^AB, P^C), each F128 [2^k_skip] on Λ.

    a/b/c_bits: uint8 [2^m] (0/1). r: uint64 [m, 2] (F128). Per row of 2^k_skip
    bits -> F8 col, inv-NTT on S then fwd-NTT on Λ, then accumulate
    eq(r[k_skip:], x) · φ₈(a·b) and · φ₈(c). Byte-identical to flock's
    `round1_naive`; equals the wire `round1_ab`/`round1_c`. (The "naive" name is
    flock's oracle reference; the compute routes through the device-fused
    `round1_rows` / `_round1_core`.)
    """
    a = witness_to_rows(a_bits, m, k_skip)
    b = witness_to_rows(b_bits, m, k_skip)
    c = witness_to_rows(c_bits, m, k_skip)
    return round1_rows(a, b, c, m, k_skip, r)
