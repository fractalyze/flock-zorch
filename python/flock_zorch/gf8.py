"""F8 = GF(2⁸) (AES field) + the φ₈ embedding into F128 + the additive NTT over
F8 — the machinery flock's zerocheck round-1 univariate-skip URM (`round1_naive`)
needs. Byte-identical to flock-core's `field/gf2_8.rs`, `field/phi8.rs`,
`ntt.rs::AdditiveNttGf8`, and `zerocheck/univariate_skip.rs::round1_naive`.

F8 work is small (64-wide columns, one pass per witness row) and one-time per
prove; byte-identity is the goal, not throughput — the bulk GPU win is the F128
multilinear rounds + NTT. So the F8 field/NTT live on host (numpy), and only the
final F128 eq-accumulation routes through the device `field.mul`.

F8 is a DIFFERENT tower from the F128 GHASH basis: AES poly x⁸+x⁴+x³+x+1 = 0x11B,
linked to F128 only through φ₈ (a field homomorphism into a subfield). The
additive NTT here is also distinct from `ntt.py`'s F128 LCH NTT: a `next_s`
twiddle recurrence with binary-heap indexing and recursive DIF/DIT butterflies.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from flock_zorch import field, sumcheck

# ---------------------------------------------------------------------------
# F8 = GF(2^8), AES irreducible x^8+x^4+x^3+x+1 = 0x11B (reduction const 0x1B).
# ---------------------------------------------------------------------------


def _clmul8(a: int, b: int) -> int:
    """Carry-less (GF(2)[x]) product of two bytes -> u16."""
    acc = 0
    for i in range(8):
        if (a >> i) & 1:
            acc ^= b << i
    return acc


def _gf8_reduce(p: int) -> int:
    """Reduce a degree-<=14 product mod 0x11B. Two folds (x^8 = x^4+x^3+x+1);
    the first fold can re-overflow bit 8, so a second pass is required."""
    h = p >> 8
    t = (p & 0xFF) ^ h ^ (h << 1) ^ (h << 3) ^ (h << 4)
    h2 = t >> 8
    return ((t & 0xFF) ^ h2 ^ (h2 << 1) ^ (h2 << 3) ^ (h2 << 4)) & 0xFF


# Full 256x256 F8 multiply table + 256-entry Fermat (x^254) inverse table, host.
_MUL = np.array(
    [[_gf8_reduce(_clmul8(a, b)) for b in range(256)] for a in range(256)],
    dtype=np.uint8,
)


def _gf8_inv_scalar(a: int) -> int:
    result, sq = 1, a
    for i in range(8):  # exponent x^254 = bits 0xFE
        if (0xFE >> i) & 1:
            result = _gf8_reduce(_clmul8(result, sq))
        sq = _gf8_reduce(_clmul8(sq, sq))
    return result


_INV = np.array([_gf8_inv_scalar(a) for a in range(256)], dtype=np.uint8)


def gf8_mul(a, b):
    """Elementwise F8 multiply (broadcasts via the 256x256 table gather)."""
    return _MUL[np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8)]


def gf8_inv(a):
    return _INV[np.asarray(a, dtype=np.uint8)]


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


PHI_8_TABLE = _build_phi8_table()  # uint64 [256, 2] = F128


def phi8(v) -> np.ndarray:
    """F8 byte(s) -> F128 (uint64[..., 2])."""
    return PHI_8_TABLE[np.asarray(v, dtype=np.uint8)]


# ---------------------------------------------------------------------------
# AdditiveNttGf8 — additive NTT over F8 (LCH novel-poly basis, coset offset beta).
# ---------------------------------------------------------------------------


def _next_s(s: int, root: int) -> int:
    """next_s(s, root) = s*s + root*s = s*(s+root), in F8."""
    return int(gf8_mul(s, s)) ^ int(gf8_mul(root, s))


def _compute_twiddles(k: int, beta: int) -> np.ndarray:
    """Binary-heap twiddle table, uint8 [2^k - 1]. Level-L twiddles at offset
    2^L - 1. Distinct from the F128 NTT's layer-major table."""
    if k == 0:
        return np.zeros(0, dtype=np.uint8)
    n = 1 << k
    twiddles = np.zeros(n - 1, dtype=np.uint8)
    length = 1 << (k - 1)
    layer = [(int(beta) ^ ((2 * i) & 0xFF)) for i in range(length)]  # beta + F8(2*i)
    s_at_root = 1
    write_at = length
    for i in range(length):  # level 0 written as-is (s_at_root = 1)
        twiddles[write_at - 1 + i] = layer[i]
    for _ in range(1, k):
        write_at >>= 1
        next_s_root = _next_s(layer[1] ^ layer[0], s_at_root)
        new_len = write_at
        layer = [_next_s(layer[2 * i], s_at_root) for i in range(new_len)]  # uses OLD s_at_root
        length = new_len
        s_at_root = next_s_root
        s_inv = int(gf8_inv(s_at_root))
        for j in range(length):
            twiddles[write_at - 1 + j] = int(gf8_mul(s_inv, layer[j]))
    return twiddles


def _fft(v: np.ndarray, tw: np.ndarray, idx: int) -> np.ndarray:
    """Decimation-in-frequency: butterfly first, then recurse on the contiguous
    halves of the last axis. tw indexed binary-heap (node idx -> tw[idx-1])."""
    if v.shape[-1] == 1:
        return v
    half = v.shape[-1] // 2
    lam = int(tw[idx - 1])
    lo, hi = v[..., :half], v[..., half:]
    new_lo = lo ^ gf8_mul(lam, hi)  # v[i] += lam*w
    new_hi = hi ^ new_lo            # v[half+i] = w + v[i] (updated)
    lo2 = _fft(new_lo, tw, 2 * idx)
    hi2 = _fft(new_hi, tw, 2 * idx + 1)
    return np.concatenate([lo2, hi2], axis=-1)


def _ifft(v: np.ndarray, tw: np.ndarray, idx: int) -> np.ndarray:
    """Decimation-in-time: recurse first, then butterfly last."""
    if v.shape[-1] == 1:
        return v
    half = v.shape[-1] // 2
    lo = _ifft(v[..., :half], tw, 2 * idx)
    hi = _ifft(v[..., half:], tw, 2 * idx + 1)
    lam = int(tw[idx - 1])
    new_hi = hi ^ lo                # v[half+i] += v[i]
    new_lo = lo ^ gf8_mul(lam, new_hi)  # v[i] += lam*v[half+i] (updated)
    return np.concatenate([new_lo, new_hi], axis=-1)


class AdditiveNttGf8:
    """Additive NTT over F8 on the coset W = beta + span{1,2,…,2^(k-1)}."""

    def __init__(self, k: int, beta: int):
        self.k = k
        self.beta = int(beta) & 0xFF
        self.twiddles = _compute_twiddles(k, self.beta)

    def forward(self, v: np.ndarray) -> np.ndarray:
        """v: uint8 [..., 2^k] -> transformed [..., 2^k]."""
        return _fft(np.asarray(v, dtype=np.uint8), self.twiddles, 1)

    def inverse(self, v: np.ndarray) -> np.ndarray:
        return _ifft(np.asarray(v, dtype=np.uint8), self.twiddles, 1)


# ---------------------------------------------------------------------------
# Device (GPU) F8: the same tables + additive NTT in jnp, so round1_naive runs
# on the GPU instead of host numpy (the URM was the prover's #1 host bottleneck:
# ~1.1 s at m=24). Byte-identical to the host path (gated by the URM oracle).
# ---------------------------------------------------------------------------

import functools  # noqa: E402

_MUL_DEV = jnp.asarray(_MUL)            # [256, 256] uint8
_PHI_DEV = jnp.asarray(PHI_8_TABLE)     # [256, 2] uint64


def _gf8_mul_dev(a, b):
    """Elementwise F8 multiply on device via the 256x256 table gather."""
    return _MUL_DEV[a.astype(jnp.int32), b.astype(jnp.int32)]


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


@functools.partial(jax.jit, static_argnums=(3,))
def _extend_and_phi(a, b, c, k_skip, tw_s, tw_l):
    """Extend a/b/c from S to Lambda (inv-NTT then fwd-NTT), F8-mul a·b, and
    phi8-embed both into F128. No field mul → jit-cacheable independent of `mul`.
    a/b/c: uint8 [N, ell] -> (phi_ab, phi_c) uint64 [N, ell, 2]."""
    a_l = _fft_dev(_ifft_dev(a, tw_s, k_skip), tw_l, k_skip)
    b_l = _fft_dev(_ifft_dev(b, tw_s, k_skip), tw_l, k_skip)
    c_l = _fft_dev(_ifft_dev(c, tw_s, k_skip), tw_l, k_skip)
    ab = _gf8_mul_dev(a_l, b_l)
    return _PHI_DEV[ab.astype(jnp.int32)], _PHI_DEV[c_l.astype(jnp.int32)]


_ACCUM_CACHE: dict = {}


def _accum(mul):
    """jit'd eq-accumulator p[s] = Σ_x eq[x]·phi[x,s], cached per `mul` (→ clmad)."""
    fn = _ACCUM_CACHE.get(mul)
    if fn is None:
        fn = jax.jit(lambda eqx, phi: sumcheck._xor_reduce(mul(eqx, phi), axis=0))
        _ACCUM_CACHE[mul] = fn
    return fn


# ---------------------------------------------------------------------------
# round1_naive — the zerocheck round-1 URM reference (== the wire round1_ab/c).
# ---------------------------------------------------------------------------


def round1_naive(a_bits, b_bits, c_bits, m: int, k_skip: int, r, mul=field.mul):
    """Round-1 univariate-skip message (P^AB, P^C), each F128 [2^k_skip] on Λ.

    a/b/c_bits: uint8 [2^m] (0/1). r: uint64 [m, 2] (F128). Per row of 2^k_skip
    bits -> F8 col, inv-NTT on S then fwd-NTT on Λ, then accumulate
    eq(r[k_skip:], x) · φ₈(a·b) and · φ₈(c). Byte-identical to flock's
    `round1_naive`; equals the wire `round1_ab`/`round1_c`.
    """
    ell = 1 << k_skip
    n_chunks = 1 << (m - k_skip)
    # F8 NTT + a·b + phi8 on the GPU (the URM was the prover's #1 host bottleneck);
    # twiddles are tiny (2^k-1 entries) so they're computed on host once.
    tw_s = jnp.asarray(_compute_twiddles(k_skip, 0))     # S = {0..ell-1}
    tw_l = jnp.asarray(_compute_twiddles(k_skip, ell))   # Lambda = {ell..2*ell-1}
    a = jnp.asarray(np.asarray(a_bits, np.uint8).reshape(n_chunks, ell))
    b = jnp.asarray(np.asarray(b_bits, np.uint8).reshape(n_chunks, ell))
    c = jnp.asarray(np.asarray(c_bits, np.uint8).reshape(n_chunks, ell))

    phi_ab, phi_c = _extend_and_phi(a, b, c, k_skip, tw_s, tw_l)  # [n_chunks, ell, 2]
    r = np.asarray(r, dtype=np.uint64)
    eqx = sumcheck.build_eq(jnp.asarray(r[k_skip:]), mul=mul)[:, None, :]  # [n_chunks, 1, 2]

    accum = _accum(mul)
    return np.asarray(accum(eqx, phi_ab)), np.asarray(accum(eqx, phi_c))
