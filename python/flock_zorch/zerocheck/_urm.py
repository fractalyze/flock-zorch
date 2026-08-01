"""φ₈ embedding of F8 = GF(2⁸) (AES field) into F128 + the zerocheck round-1
univariate-skip URM (`round1_rows`) — host orchestration and the fused device
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

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import lax

from flock_zorch import ghash, sumcheck

# ---------------------------------------------------------------------------
# phi8: F8 -> F128 embedding (256-entry table). F2-linear, so the full table is
# built by XOR over set bits from the 8 basis images phi8(2^t). Pinned to
# flock's PHI_8_TABLE transitively by the proof-level byte gates.
# ---------------------------------------------------------------------------

_PHI8_BASIS = np.array(
    [
        [0x0000000000000001, 0x0000000000000000],  # phi8(0x01)
        [0x6B8330483C2E9849, 0x0DCB364640A222FE],  # phi8(0x02)
        [0x7573DA4A5F7710ED, 0x3D5BD35C94646A24],  # phi8(0x04)
        [0x41A12DB1F974F3AC, 0x6D58C4E181F9199F],  # phi8(0x08)
        [0x5E2F716F4EDE412F, 0xA72EC17764D7CED5],  # phi8(0x10)
        [0x5CB10FBABCF00118, 0x4D52354A3A3D8C86],  # phi8(0x20)
        [0x95ED1F57F3632D4D, 0x553E92E8BC0AE9A7],  # phi8(0x40)
        [0x512625B1F09FA87E, 0x93252331BF042B11],  # phi8(0x80)
    ],
    dtype=np.uint64,
)


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
_PHI_DEV_G = ghash.to_ghash(
    _PHI_DEV
)  # [256] ghash — indexed in-kernel, no lane bitcast
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


# Rows per block once round-1 blocks (see `_round1_core`). 2**22 is the row
# count m=28 runs at unblocked, so a block reproduces a working set the stack is
# already measured on: 256 MiB per track, ~1.8 GiB for the whole core.
_ROUND1_BLOCK_ROWS = 1 << 22


def _round1_partial(a, b, c, eqx, k_skip):
    """One block's contribution to (P^AB, folded-C): extend a/b S→Λ, a·b,
    φ8-embed and eq-accumulate over this block's rows. Fused, so the large
    [rows, ell] φ8 intermediate is consumed in-fusion and never written to HBM
    (halves round1's bandwidth vs a separate extend + accumulate).

    The C track is only FOLDED here — `Σ_x eq_x · c_x[s]`, a select-XOR since
    the rows are bits — and its S→Λ extension happens once on the accumulated
    result in `_round1_core` (`_extend_folded_c`). For C the two orders are
    equal: the extend is linear and φ8 is a homomorphism, so
    `Σ_x eq_x·φ8(LDE(c_x)) = LDE_F128(Σ_x eq_x·c_x)`. AB cannot reorder —
    `a·b` is a product, formed pointwise on the extended domain — which is why
    only C drops its per-row extend, φ8 gather and clmul accumulate."""
    a_l = _extend_rows(a, k_skip)
    b_l = _extend_rows(b, k_skip)
    ab = _to_u8(a_l * b_l).astype(fnp.int32)
    phi_ab = _PHI_DEV_G[ab]
    c_sel = fnp.where(c.astype(bool), eqx, fnp.zeros((), fnp.binary_field_ghash))
    return fnp.sum(eqx * phi_ab, axis=0), fnp.sum(c_sel, axis=0)


def _extend_folded_c(v, k_skip: int):
    """S→Λ extension of the folded C vector, in F128: `P^C(λ) = Σ_s v[s] ·
    φ8(LDE(e_s)(λ))`. The `[ell, ell]` basis matrix is built by running the SAME
    row extend the per-row path used on the identity — so the twiddle/coset
    convention cannot drift from `_extend_rows` — and φ8-embedding it. A tiny
    fixed-size contraction (64×64 clmuls at k_skip=6); XLA constant-folds the
    basis."""
    ell = 1 << k_skip
    eye = fnp.asarray(np.eye(ell, dtype=np.uint8))
    basis = _PHI_DEV_G[_to_u8(_extend_rows(eye, k_skip)).astype(fnp.int32)]
    return fnp.sum(basis * v[:, None], axis=0)


@functools.partial(frx.jit, static_argnums=(3,))
def _round1_core(a, b, c, k_skip, r):
    """Fused round-1 core: build eqx, then accumulate `_round1_partial` over the
    row axis. `build_eq` is in-kernel (no `build_eq_fused`).

    **Blocked over rows above `_ROUND1_BLOCK_ROWS`.** Round-1 is a map-reduce:
    `_extend_rows` transforms along the LAST axis (ell = 2^k_skip), so rows are
    independent, and the reduction is over the row axis down to `[ell]` — 64
    ghash elements, 1 KiB, out of a 4 GiB-per-track input at m=32. Holding all
    N rows live is therefore a property of the traced program, not of the
    algorithm, and it is what puts round-1 at 28.06 GiB on a 32 GB card at m=32
    (a 16.06 GiB temp arena plus three 4 GiB `u8[N, ell]` tracks) — the OOM in
    #179. Blocking divides the arena by the block count.

    Bit-identical, not merely close: the accumulation is `+` on
    `binary_field_ghash`, i.e. XOR, which is associative and commutative, so
    partitioning the rows cannot change the result — the same argument
    `sumcheck.build_eq_suffix_tables` relies on.

    One block (N <= block rows) takes the unblocked path unchanged, so every
    instance that already fits keeps its exact program, not a scan of length 1.

    The C track folds first and extends once (`_round1_partial` /
    `_extend_folded_c`): its per-row S→Λ NTT passes, φ8 gather and clmul
    accumulate are replaced by one select-XOR reduce plus a 64-point extension
    of the reduced vector. Equal by linearity — valid for ANY c rows, not an
    identity-C special case.
    """
    eqx = sumcheck.build_eq(r[k_skip:])[:, None]  # r is ghash [m]; [n_chunks, 1]
    n_rows, ell = a.shape
    n_blocks = n_rows // _ROUND1_BLOCK_ROWS
    if n_blocks <= 1:
        p_ab, folded_c = _round1_partial(a, b, c, eqx, k_skip)
        return p_ab, _extend_folded_c(folded_c, k_skip)

    rows = _ROUND1_BLOCK_ROWS
    blocked = lambda x, w: x.reshape(n_blocks, rows, w)  # noqa: E731

    def step(acc, xs):
        part = _round1_partial(*xs[:3], xs[3], k_skip)
        return (acc[0] + part[0], acc[1] + part[1]), None

    zero = fnp.zeros(ell, fnp.binary_field_ghash)
    (acc_ab, acc_c), _ = lax.scan(
        step,
        (zero, zero),
        (blocked(a, ell), blocked(b, ell), blocked(c, ell), blocked(eqx, 1)),
    )
    return acc_ab, _extend_folded_c(acc_c, k_skip)


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
    bits = fnp.concatenate([lo, hi], axis=1).reshape(-1)  # [2^m]
    return bits.reshape(1 << (m - k_skip), 1 << k_skip)


# ---------------------------------------------------------------------------
# The zerocheck round-1 URM message (== the wire round1_ab/c).
# ---------------------------------------------------------------------------


def witness_to_rows(bits, m: int, k_skip: int):
    """Witness -> device uint8 rows [2^(m-k_skip), 2^k_skip], for round1 + fold_at_z.

    Accepts three forms: the **packed F128** witness (uint64 [2^(m-7), 2]) — unpacked
    on device (8x less host transfer, the preferred form); a uint8 [2^m] (0/1) bit
    array (transferred once); or an already-device array (reshaped, no copy)."""
    n_chunks, ell = 1 << (m - k_skip), 1 << k_skip
    if (
        getattr(bits, "ndim", 0) == 2
        and bits.shape[-1] == 2
        and np.dtype(bits.dtype) == np.uint64
    ):
        return _packed_to_rows(
            fnp.asarray(bits), m, k_skip
        )  # packed F128 -> device unpack
    if isinstance(bits, frx.Array):
        return bits.reshape(n_chunks, ell)
    return fnp.asarray(np.asarray(bits, np.uint8).reshape(n_chunks, ell))


def round1_rows(a, b, c, m: int, k_skip: int, r):
    """Round-1 univariate-skip message (P^AB, P^C), each F128 [2^k_skip] on Λ,
    from device witness rows (uint8 [2^(m-k_skip), 2^k_skip]) — split from
    `witness_to_rows` so the witness is transferred once and reused by
    `zerocheck._fold_at_z`. Per row of 2^k_skip bits -> F8 col, inv-NTT on S then
    fwd-NTT on Λ, then accumulate eq(r[k_skip:], x) · φ₈(a·b) and · φ₈(c).
    Byte-identical to flock's `round1_naive` (== the wire `round1_ab`/`round1_c`).
    Returns (P^AB, P^C) as device-resident `binary_field_ghash [2^k_skip]` — no
    host lift; consumers observe/interpolate natively and byte-gate readers
    normalize via `ghash.to_lanes`."""
    return _round1_core(a, b, c, k_skip, r)  # eqx build + extend+phi+accum, fused
