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
from zorch.fusion import fused_region
from zorch.poly.eq import expand_eq_to_hypercube

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
_PHI_BASIS_DEV_G = ghash.to_ghash(fnp.asarray(_PHI8_BASIS))
_AES = np.dtype(zk_dtypes.binary_field_gf8_aes)
URM_MARKER = "zorch.zerocheck_urm"
_ROUND1_PARTIALS = 8192


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


def is_packed_witness(bits) -> bool:
    """Is this the packed F128 witness (uint64 [2^(m-7), 2]) rather than bits?

    The one authority on that question: round-1, the multilinear fold and the
    block reshape all branch on it, and open-coding the three clauses per site
    is how they drift. `getattr` rather than `.ndim` so a host list or a scalar
    answers False instead of raising."""
    return (
        getattr(bits, "ndim", 0) == 2
        and bits.shape[-1] == 2
        and np.dtype(bits.dtype) == np.uint64
    )


def _lsb_bits(x, width: int):
    """Expand each element of unsigned-integer `x` into its `width` LSB-first
    bits, as a trailing axis of 0/1 in x's own dtype.

    The witness bit order in one place. Packed storage reaches round-1 in three
    guises — the uint8 byte planes `lax.composite` exposes, the uint64 F128
    lanes, and the a·b product bytes — and every one of them unpacks through
    here, so they cannot drift. Lazy: whether the expansion lands in HBM is the
    consumer's business, not this function's."""
    return (x[..., None] >> fnp.arange(width, dtype=x.dtype)) & fnp.ones((), x.dtype)


def _round1_input_rows(x, n_rows: int):
    """Normalize an unpacked row matrix or a physically represented packed
    F128 witness to one bit per byte for the portable decomposition.

    ``lax.composite`` exposes a logical ``uint64[..., 2]`` operand to its
    decomposition as little-endian ``uint8[..., 2, 8]`` storage.  The custom
    GPU emitter consumes those bytes directly; spelling the unpack here keeps
    the marker's fallback semantics shape-correct without materializing this
    expansion on the custom path.
    """
    if x.ndim == 1 and np.dtype(x.dtype) == np.uint8:
        if x.shape[0] == n_rows * 64:
            return x.reshape(n_rows, 64)
        assert x.shape[0] == n_rows * 8
        return _lsb_bits(x, 8).reshape(-1, 64)
    if x.ndim == 3 and x.shape[-2:] == (2, 8):
        return _lsb_bits(x, 8).reshape(-1, 64)
    if is_packed_witness(x):
        lo = _lsb_bits(x[:, 0], 64).astype(fnp.uint8)
        hi = _lsb_bits(x[:, 1], 64).astype(fnp.uint8)
        return fnp.concatenate([lo, hi], axis=1).reshape(-1, 64)
    return x


# Rows per block once round-1 blocks (see `_round1_core`). 2**22 is the row
# count m=28 runs at unblocked, so a block reproduces a working set the stack is
# already measured on: 256 MiB per track, ~1.8 GiB for the whole core.
_ROUND1_BLOCK_ROWS = 1 << 22


def _round1_partial_decomp(
    a, b, c, eq_or_point, phi_basis, *, k_skip: int, with_c: bool = True
):
    """Portable decomposition of the fused bit-sliced URM map-reduce.

    The custom GPU emitter keeps the transformed AES bit planes on chip and
    emits only GHASH partials.  This decomposition spells the same operation in
    ordinary array primitives for CPU and marker fallback.

    `with_c=False` (the identity-C stripe shortcut, #192) skips the C track and
    returns only the P^AB partial; the AB arithmetic is untouched, so P^AB is
    bit-identical either way.
    """

    def matches_rows(rows: int) -> bool:
        if a.ndim == 3 and a.shape[-2:] == (2, 8):
            return a.shape[0] * 2 == rows
        if is_packed_witness(a):
            return a.shape[0] * 2 == rows
        if a.ndim == 1 and np.dtype(a.dtype) == np.uint8:
            return a.shape[0] in (rows * 8, rows * 64)
        return a.shape[0] == rows

    point_rows = 1 << eq_or_point.shape[0]
    point_weights = eq_or_point.shape[0] < 63 and matches_rows(point_rows)
    eqx = (
        expand_eq_to_hypercube(eq_or_point, fnp.ones((), eq_or_point.dtype), msb=True)
        if point_weights
        else eq_or_point.reshape(-1)
    )
    n_rows = eqx.shape[0]
    a = _round1_input_rows(a, n_rows)
    b = _round1_input_rows(b, n_rows)
    a_l = _extend_rows(a, k_skip)
    b_l = _extend_rows(b, k_skip)
    byte_values = _to_u8(a_l * b_l)
    n_partials = min(_ROUND1_PARTIALS, n_rows)
    rows_per_partial = n_rows // n_partials
    selected = _lsb_bits(byte_values, 8).astype(fnp.bool_)
    selected = selected.reshape(n_partials, rows_per_partial, 64, 8)
    weights = eqx.reshape(n_partials, rows_per_partial, 1, 1)
    zero = ghash.to_ghash(fnp.zeros((2,), dtype=fnp.uint64))
    bit_sums = fnp.sum(fnp.where(selected, weights, zero), axis=1)
    partial_ab = fnp.sum(bit_sums * phi_basis[None, None, :], axis=2)
    if not with_c:
        return partial_ab  # [n_partials, ell] — stripe shortcut, no C track

    c = _round1_input_rows(c, n_rows)
    c_selected = c.astype(fnp.bool_).reshape(n_partials, rows_per_partial, 64)
    c_weights = eqx.reshape(n_partials, rows_per_partial, 1)
    partial_c = fnp.sum(fnp.where(c_selected, c_weights, zero), axis=1)
    return fnp.stack([partial_ab, partial_c], axis=1)


def _round1_partial(a, b, c, eq_or_point, k_skip, with_c=True):
    """One row block's canonical zerocheck URM composite. `with_c=False` (the
    identity-C stripe shortcut, #192) bypasses the fused marker and runs the
    portable decomposition AB-only, so the C track is never computed on device."""
    if not with_c:
        partial_ab = _round1_partial_decomp(
            a,
            b,
            None,
            eq_or_point.reshape(-1),
            _PHI_BASIS_DEV_G,
            k_skip=k_skip,
            with_c=False,
        )
        return fnp.sum(partial_ab, axis=0)
    partials = fused_region(
        _round1_partial_decomp,
        a,
        b,
        c,
        eq_or_point.reshape(-1),
        _PHI_BASIS_DEV_G,
        name=URM_MARKER,
        version=1,
        k_skip=k_skip,
    )
    out = fnp.sum(partials, axis=0)
    return out[0], out[1]


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


@functools.partial(frx.jit, static_argnums=(3, 5))
def _round1_core(a, b, c, k_skip, r, with_c=True):
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
    outer_point = r[k_skip:]
    n_rows, ell = 1 << outer_point.shape[0], 1 << k_skip
    n_blocks = n_rows // _ROUND1_BLOCK_ROWS
    if n_blocks <= 1:
        part = _round1_partial(a, b, c, outer_point, k_skip, with_c)
        return (part[0], _extend_folded_c(part[1], k_skip)) if with_c else part

    eqx = sumcheck.build_eq(outer_point)[:, None]

    rows = _ROUND1_BLOCK_ROWS
    packed = is_packed_witness(a)

    def blocked_witness(x):
        if packed:
            # One packed F128 element holds two consecutive 64-bit rows.
            return x.reshape(n_blocks, rows // 2, 2)
        return x.reshape(n_blocks, rows, ell)

    eqx_blocks = eqx.reshape(n_blocks, rows, 1)
    zero = fnp.zeros(ell, fnp.binary_field_ghash)

    # `with_c=False` (stripe shortcut, #192) drops the C track; the block-scan
    # scaffold is shared so it cannot drift from the drain path.
    if not with_c:

        def step_ab(acc, xs):
            return (
                acc + _round1_partial(xs[0], xs[1], None, xs[2], k_skip, False),
                None,
            )

        acc_ab, _ = lax.scan(
            step_ab, zero, (blocked_witness(a), blocked_witness(b), eqx_blocks)
        )
        return acc_ab

    def step(acc, xs):
        part = _round1_partial(*xs[:3], xs[3], k_skip)
        return (acc[0] + part[0], acc[1] + part[1]), None

    (acc_ab, acc_c), _ = lax.scan(
        step,
        (zero, zero),
        (blocked_witness(a), blocked_witness(b), blocked_witness(c), eqx_blocks),
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
    lo = _lsb_bits(packed[:, 0], 64).astype(fnp.uint8)
    hi = _lsb_bits(packed[:, 1], 64).astype(fnp.uint8)
    bits = fnp.concatenate([lo, hi], axis=1).reshape(-1)  # [2^m]
    return bits.reshape(1 << (m - k_skip), 1 << k_skip)


# ---------------------------------------------------------------------------
# The zerocheck round-1 URM message (== the wire round1_ab/c).
# ---------------------------------------------------------------------------


def witness_to_rows(bits, m: int, k_skip: int):
    """Witness -> device uint8 rows [2^(m-k_skip), 2^k_skip], for round1 + fold_at_z.

    Accepts three forms: the **packed F128** witness (uint64 [2^(m-7), 2]) — unpacked
    on device (8x less host transfer, the preferred form); a uint8 [2^m] (0/1) bit
    array (transferred once); or an already-device array, reshaped eagerly. That
    last reshape is NOT free: outside a trace it dispatches its own program and
    allocates a fresh buffer, so it copies the whole witness."""
    n_chunks, ell = 1 << (m - k_skip), 1 << k_skip
    if is_packed_witness(bits):
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


# ---------------------------------------------------------------------------
# The identity-C shortcut (flock-zorch#192).
#
# When C is the identity (Cz = z), the round-1 C message carries no information
# the lincheck stripe does not already hold: both are partial folds of the SAME
# witness z, over the SAME high bits. So instead of draining C from the
# row-major witness (an extra 2^m-scale track), we compute only P^AB
# (`round1_ab_rows` = `_round1_core` with `with_c=False`, which bypasses the URM
# marker so the C track is never computed on device) and DERIVE P^C from the
# stripe (`round1_c_from_stripe`). Byte-identical by construction — proven in
# `stripe_c_test`.
# ---------------------------------------------------------------------------


def round1_ab_rows(a, b, k_skip: int, r):
    """Round-1 P^AB only (`binary_field_ghash [2^k_skip]` on Λ). The C track is
    taken from the lincheck stripe by `round1_c_from_stripe`, so the C witness
    never enters round-1. Byte-identical to `round1_rows(...)[0]`."""
    return _round1_core(a, b, None, k_skip, r, with_c=False)


@functools.partial(frx.jit, static_argnums=(2,))
def _round1_c_from_zvec(z_vec, r_mid, k_skip):
    """Given the inner table `z_vec` (the stripe folded at r_outer, length
    2^k_log), fold its top (k_log − k_skip) bits at r_mid = r[k_skip:k_log] to
    the length-2^k_skip S-domain vector `folded_c`, then S→Λ extend it (the SAME
    `_extend_folded_c` the drain used). `z_vec[y·2^k_skip + s]`: s = low k_skip
    bits, y = the middle bits, so the reshape puts y on axis 0."""
    eq_mid = sumcheck.build_eq(r_mid)  # [2^(k_log-k_skip)]
    grid = z_vec.reshape(eq_mid.shape[0], 1 << k_skip)  # [y, s]
    folded_c = fnp.sum(eq_mid[:, None] * grid, axis=0)  # [2^k_skip] ghash
    return _extend_folded_c(folded_c, k_skip)


def round1_c_from_stripe(z_stripe, m: int, k_log: int, k_skip: int, r):
    """Derive the exact round-1 C message from the lincheck stripe instead of
    draining the row-major witness. C is identity (Cz = z), and both the drain
    (`folded_c[s] = Σ_x eq(r[k_skip:], x)·z[x·2^k_skip + s]`) and the stripe are
    partial folds of the same z. Fold the stripe at r_outer = r[k_log:] to the
    tiny 2^k_log inner table, then fold its top (k_log − k_skip) bits at
    r[k_skip:k_log] — algebraically identical to the drain by the tensor
    factorization eq(r[k_skip:]) = eq(r[k_skip:k_log]) ⊗ eq(r[k_log:]). Returns
    `binary_field_ghash [2^k_skip]` on Λ, byte-identical to `round1_rows(...)[1]`."""
    # Lazy import: lincheck.prover imports zerocheck._lagrange_weights at module
    # load, so importing it at module top here would cycle.
    from flock_zorch.lincheck.prover import partial_fold_packed_z

    z_vec = partial_fold_packed_z(z_stripe, m, k_log, r[k_log:])  # [2^k_log] ghash
    return _round1_c_from_zvec(z_vec, r[k_skip:k_log], k_skip)
