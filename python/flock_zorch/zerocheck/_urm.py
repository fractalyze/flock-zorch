"""φ₈ embedding of F8 = GF(2⁸) (AES field) into F128 + the zerocheck round-1
univariate-skip URM (`round1_rows`) — host orchestration and the fused device
core. Byte-identical to flock-core's `field/phi8.rs` and
`zerocheck/univariate_skip.rs::round1_naive`.

F8 arithmetic and its additive NTT are compiler-native: the
`binary_field_gf8_aes` dtype (AES poly x⁸+x⁴+x³+x+1 = 0x11B) dispatches the
field-generic LCH14 additive NTT through `lax.ntt`, so this module carries no
field code — only φ₈ (a field homomorphism into an F128 subfield, the only
link between the AES basis and the GHASH basis) and the round-1 plumbing.

`round1_rows` picks the formulation, not the message: the composite
(`_round1_partial_decomp`) spells the naive per-row form the GPU emitter
consumes, and `_round1_core_factored` spells the same message over the
factored row-eq — gf8 bytes throughout, one GHASH multiply per outer chunk.
Both produce the identical round-1 message.

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
from hash_frx.fusion import fused_region
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
# Partials count for the round-1 map-reduce, per eq form (measured m28/m32,
# RTX 5090, ptxas 13.3, on the pre-window blocked scan — the counts are
# per-launch and carry over): with the eq block pre-materialized 16384 is
# faster — 833 -> 792 us per `_ROUND1_BLOCK_ROWS` rows (the two constants were
# measured together) against a +7 us partials-sum, ~ -0.5 ms on the m32 URM
# window (flock-zorch#200) — while the point-form composite (eq built
# in-fusion) regresses 915 -> 1010 us at the same doubling. Byte-safe for any
# count by `_round1_core`'s partition argument.
_ROUND1_PARTIALS_POINT = 8192
_ROUND1_PARTIALS_EQX = 16384


# ---------------------------------------------------------------------------
# Device (GPU) round-1 URM core.
# ---------------------------------------------------------------------------


def _extend_rows(rows, k_skip: int):
    """S→Λ extension, uint8 rows [N, 2^k_skip] -> AES-dtype rows on Λ."""
    ell = 1 << k_skip
    v = lax.bitcast_convert_type(rows, _AES)
    coeffs = lax.ntt(v, ntt_type="INTT", ntt_length=ell)
    return lax.ntt(coeffs, ntt_type="NTT", ntt_length=ell, coset=ell)


# The circulant byte table extends a row of BITS, one 64-bit word per 8 λ, so
# it covers exactly k_skip = 6 on the packed witness. `_extend_rows` stays the
# path for every other row form: raw F8 bytes, an unpacked bit array, any other
# k_skip.
_EXT_TABLE_K_SKIP = 6
_EXT_WORDS = 8  # 64-bit words per extended row at that k_skip


@functools.cache
def _t0word() -> np.ndarray:
    """`ext(v at byte group 0)` packed 8 bytes per word — uint64 [256, 8].

    Row v is the S→Λ extension of the ℓ = 64 row carrying the bits of byte v in
    columns 0..7 and zero elsewhere, with word w holding extension bytes
    8w .. 8w + 7 little-endian. Built by running `_extend_rows` itself on those
    256 rows, so the twiddle/coset convention cannot drift from the per-row
    path.

    One row group generates every 64-bit row, by the circulant
    `ext(v at group b)[8w + i] == ext(v at 0)[8(w ^ b) + i]`: XOR the eight
    table words, then extract one byte per λ. Both tiers read this table — the
    Triton kernel through `_urm_pallas._tables`, XLA:CPU through
    `_extend_packed_rows`.
    """
    byte_rows = np.zeros((256, 1 << _EXT_TABLE_K_SKIP), dtype=np.uint8)
    byte_rows[:, :8] = (np.arange(256)[:, None] >> np.arange(8)) & 1
    # Concrete even under an enclosing trace: the table is an input-independent
    # constant (the @cache would otherwise capture tracers).
    with frx.ensure_compile_time_eval():
        ext = np.asarray(
            _to_u8(_extend_rows(fnp.asarray(byte_rows), _EXT_TABLE_K_SKIP))
        )
    t0word = np.zeros((256, _EXT_WORDS), dtype=np.uint64)
    for i in range(8):
        t0word |= ext[:, i::8].astype(np.uint64) << np.uint64(8 * i)
    return t0word


@functools.cache
def _circulant_words() -> np.ndarray:
    """`_t0word()` pre-permuted per byte group — uint64 [8 * 256, 8], indexed
    `b * 256 + v` (the flat convention `_CONVERT_DEV_G` uses) with entry
    `[b * 256 + v][w] == t0word[v][w ^ b]`.

    Folding the circulant's XOR into the table leaves the eight groups reading
    one table at eight offsets, which is what lets them be ONE gather. The
    Triton kernel XORs the index at runtime instead, because there the thing
    worth keeping distinct is the gather lane."""
    w = np.arange(_EXT_WORDS)
    return np.concatenate([_t0word()[:, w ^ b] for b in range(_EXT_WORDS)])


def _extend_packed_rows(rows):
    """S→Λ extension of packed 64-bit witness rows: uint64 [N] -> uint8 [N, 64].

    The circulant form of `_extend_rows` at `_EXT_TABLE_K_SKIP`, and the reason
    the CPU round-1 runs no NTT: a row is 8 byte groups, so its extension is the
    XOR of 8 table words plus one byte extract per λ, where the NTT form pays a
    whole INTT + coset NTT per row.

    Reference — flock's `InvNttTableByteSingleGf8`, the same table and the same
    XOR-shift loop, one byte per element where this packs eight per word:
      https://github.com/succinctlabs/flock/blob/85fc0e7cc002e7ca4dffdff805ba89976e9a5293/crates/flock-core/src/ntt/inv_table.rs#L154-L167
    and its §2.1 derivation of the relation:
      https://github.com/succinctlabs/flock/blob/85fc0e7cc002e7ca4dffdff805ba89976e9a5293/crates/flock-core/src/ntt/inv_table.rs#L1-L21

    **Spelled as one gather under an XOR-reduce, not eight gathers XORed.** They
    compute the same words, but XLA:CPU fuses the gather into the reduce and
    materializes only the [N, 8] result, where the eight-gather form emits eight
    separate fusions and writes then re-reads a whole [N, 8] buffer per group —
    enough extra traffic to lose to the NTT this replaces."""
    table = fnp.asarray(_circulant_words())
    group = np.arange(_EXT_WORDS)
    # The same vector splits both ends: a source row into its 8 byte groups,
    # then each result word into the 8 λ bytes it carries.
    byte_shift = fnp.asarray((group * 8).astype(np.uint64))
    offset = fnp.asarray((group * 256).astype(np.int32))
    src = ((rows[:, None] >> byte_shift) & np.uint64(0xFF)).astype(fnp.int32)
    words = lax.reduce(table[offset + src], np.uint64(0), lax.bitwise_xor, (1,))
    return (
        ((words[:, :, None] >> byte_shift) & np.uint64(0xFF))
        .astype(fnp.uint8)
        .reshape(-1, _EXT_WORDS * 8)
    )


def _to_u8(x):
    return lax.bitcast_convert_type(x, fnp.uint8)


def is_packed_witness(bits) -> bool:
    """Is this the packed F128 witness (uint64 [2^(m-7), 2]) rather than bits?

    The one authority on that question: round-1's input normalize, the
    multilinear fold and `witness_to_rows` all branch on it, and open-coding
    the three clauses per site is how they drift. `getattr` rather than `.ndim`
    so a host list or a scalar answers False instead of raising."""
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


# Rows one eq window covers once round-1 goes windowed, and — deliberately the
# same number — the row count above which round-1 windows at all (see
# `_round1_core`). 2**22 is the row count m=28 runs unwindowed, so a window
# reproduces a working set the stack is already measured on. Retuning the
# window size therefore also moves the cutoff; split the two if that stops
# being what you want.
_ROUND1_BLOCK_ROWS = 1 << 22


def _round1_partial_decomp(
    a, b, c, eq_or_point, phi_basis, *, k_skip: int, rows: int | None = None
):
    """Portable decomposition of the fused bit-sliced URM map-reduce.

    The custom GPU emitter keeps the transformed AES bit planes on chip and
    emits only GHASH partials.  This decomposition spells the same operation in
    ordinary array primitives for CPU and marker fallback.

    `rows` (a composite attribute, so it rides to this fallback) selects the
    window form: the weights operand covers only `rows / len(weights)` periods
    and tiles across the witness — the contract the GPU emitter implements as
    `weights[row mod window]`. Without it the weights either span every row or
    are a challenge point, and the row count is derived as before.
    """

    def matches_rows(n: int) -> bool:
        if a.ndim == 3 and a.shape[-2:] == (2, 8):
            return a.shape[0] * 2 == n
        if is_packed_witness(a):
            return a.shape[0] * 2 == n
        if a.ndim == 1 and np.dtype(a.dtype) == np.uint8:
            return a.shape[0] in (n * 8, n * 64)
        return a.shape[0] == n

    # The point form is a version-1 shape only: under the window contract the
    # weights operand is an eq window by definition, so `rows` answers the
    # question the shape heuristic would otherwise have to guess at.
    point_rows = 1 << eq_or_point.shape[0]
    point_weights = (
        rows is None and eq_or_point.shape[0] < 63 and matches_rows(point_rows)
    )
    eqx = (
        expand_eq_to_hypercube(eq_or_point, fnp.ones((), eq_or_point.dtype), msb=True)
        if point_weights
        else eq_or_point.reshape(-1)
    )
    if rows is not None:
        assert rows % eqx.shape[0] == 0, "window must divide the row count"
        eqx = fnp.tile(eqx, rows // eqx.shape[0])
    n_rows = eqx.shape[0]
    a = _round1_input_rows(a, n_rows)
    b = _round1_input_rows(b, n_rows)
    c = _round1_input_rows(c, n_rows)
    a_l = _extend_rows(a, k_skip)
    b_l = _extend_rows(b, k_skip)
    byte_values = _to_u8(a_l * b_l)
    partials = _ROUND1_PARTIALS_POINT if point_weights else _ROUND1_PARTIALS_EQX
    n_partials = min(partials, n_rows)
    rows_per_partial = n_rows // n_partials
    selected = _lsb_bits(byte_values, 8).astype(fnp.bool_)
    selected = selected.reshape(n_partials, rows_per_partial, 64, 8)
    weights = eqx.reshape(n_partials, rows_per_partial, 1, 1)
    zero = ghash.to_ghash(fnp.zeros((2,), dtype=fnp.uint64))
    bit_sums = fnp.sum(fnp.where(selected, weights, zero), axis=1)
    partial_ab = fnp.sum(bit_sums * phi_basis[None, None, :], axis=2)

    c_selected = c.astype(fnp.bool_).reshape(n_partials, rows_per_partial, 64)
    c_weights = eqx.reshape(n_partials, rows_per_partial, 1)
    partial_c = fnp.sum(fnp.where(c_selected, c_weights, zero), axis=1)
    return fnp.stack([partial_ab, partial_c], axis=1)


def _round1_partials(a, b, c, eq_or_point, k_skip, rows: int | None = None):
    """The fused URM composite, as raw `[n_partials, 2, ell]` partial sums.

    Partial p covers the contiguous row range `[p·rows_per_partial,
    (p+1)·rows_per_partial)` — part of the contract, since a windowed caller
    regroups this axis by block.

    `rows` widens the call to the window contract (version 2); the weights
    operand is then an eq window, per `_round1_partial_decomp`."""
    contract = {"version": 1} if rows is None else {"version": 2, "rows": rows}
    return fused_region(
        _round1_partial_decomp,
        a,
        b,
        c,
        eq_or_point.reshape(-1),
        _PHI_BASIS_DEV_G,
        name=URM_MARKER,
        k_skip=k_skip,
        **contract,
    )


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
    """Fused round-1 core: build the eq weights, then reduce the composite's
    partial sums over the row axis. `build_eq` is in-kernel (no
    `build_eq_fused`).

    **Windowed over rows above `_ROUND1_BLOCK_ROWS`.** Round-1 is a map-reduce:
    `_extend_rows` transforms along the LAST axis (ell = 2^k_skip), so rows are
    independent, and the reduction is over the row axis down to `[ell]` — 64
    ghash elements, 1 KiB. Above the threshold ONE composite launch covers
    every row under the version-2 window contract; the mechanism and why it is
    bit-identical are in the inline comment below. Every instance that already
    fits keeps its exact program (threshold rationale at
    `_ROUND1_BLOCK_ROWS`).

    The C track folds first and extends once (`_round1_partials` /
    `_extend_folded_c`): its per-row S→Λ NTT passes, φ8 gather and clmul
    accumulate are replaced by one select-XOR reduce plus a 64-point extension
    of the reduced vector. Equal by linearity — valid for ANY c rows, not an
    identity-C special case.
    """
    outer_point = r[k_skip:]
    n_rows, ell = 1 << outer_point.shape[0], 1 << k_skip
    n_blocks = n_rows // _ROUND1_BLOCK_ROWS
    if n_blocks <= 1:
        out = fnp.sum(_round1_partials(a, b, c, outer_point, k_skip), axis=0)
        return out[0], _extend_folded_c(out[1], k_skip)

    # eq factors across the block split: `build_eq` pairs challenge i with row
    # bit i, and a block is the high bits of the row index, so
    # `eqx[blk·rows + j] = row_eq[j] · block_eq[blk]`. The composite's window
    # contract (version 2) reads `row_eq[row mod window]` on chip, so ONE
    # launch covers every block with one shared table — no per-block witness
    # slices (the 16-step scan ran them as 96 MiB of DtoD per step,
    # flock-zorch#200's "non-kernel" bucket) and no full-eqx build. The block
    # scalar then lands on the per-block partial sums. Bit-identical, not
    # merely close: accumulation is `+` on `binary_field_ghash`, i.e. XOR, so
    # regrouping rows is free, and the GHASH multiply distributes exactly over
    # it — `s·Σ where(sel, t, 0)·φ == Σ where(sel, s·t, 0)·φ` — so the message
    # bytes cannot move. The same argument `sumcheck.build_eq_suffix_tables`
    # relies on.
    n_block_bits = n_blocks.bit_length() - 1
    row_eq = sumcheck.build_eq(outer_point[:-n_block_bits])
    block_eq = sumcheck.build_eq(outer_point[-n_block_bits:])

    partials = _round1_partials(a, b, c, row_eq, k_skip, rows=n_rows)
    # Regrouping partials by block requires whole partials per block, i.e. at
    # least as many partials as blocks (both are powers of two).
    assert partials.shape[0] % n_blocks == 0
    per_block = fnp.sum(partials.reshape(n_blocks, -1, 2, ell), axis=1)
    acc = fnp.sum(per_block * block_eq[:, None, None], axis=0)
    return acc[0], _extend_folded_c(acc[1], k_skip)


# ---------------------------------------------------------------------------
# Factored-eq round-1 core — the CPU tier's formulation.
# ---------------------------------------------------------------------------

# The protocol's inner dims, split by the identity each one admits. The three
# small challenges are φ₈([0xF7, 0x53, 0xB5]) and the four medium ones are
# β_i = γ^(2^(i - 1)) / (1 + γ^(2^(i - 1))), so `prover.N_INNER == 3 + 4` here.
_N_SMALL, _N_MEDIUM = 3, 4
_SMALL, _MEDIUM = 1 << _N_SMALL, 1 << _N_MEDIUM


def _gf8_reduce(p):
    """AES-poly (0x11B) reduce of a value below 2¹⁶, elementwise.

    x⁸ ≡ 0x1B, so folding the high bits is `h ^ (h << 1) ^ (h << 3) ^ (h << 4)`;
    two folds suffice because the first leaves at most 11 bits. Shared with the
    Triton kernel, which spells the same reduction on int32 lanes."""
    h = p >> 8
    t = (p & 0xFF) ^ h ^ (h << 1) ^ (h << 3) ^ (h << 4)
    h2 = t >> 8
    return (t & 0xFF) ^ h2 ^ (h2 << 1) ^ (h2 << 3) ^ (h2 << 4)


def _build_convert_table() -> np.ndarray:
    """`γᵇ · φ₈(v)` for b ∈ [0, 16), v ∈ [0, 256) — uint64 `[16, 256, 2]` lanes.

    flock's `univariate_skip_optimized.rs::build_convert_table`, 64 KB. γ is
    GHASH's x, so the doubling chain is the lane shift with the 0x87 carry
    reduction and needs no field multiply to build."""
    table = np.zeros((_MEDIUM, 256, 2), dtype=np.uint64)
    table[0] = PHI_8_TABLE
    for b in range(1, _MEDIUM):
        lo, hi = table[b - 1, :, 0], table[b - 1, :, 1]
        carry = np.where(hi >> np.uint64(63) != 0, np.uint64(0x87), np.uint64(0))
        table[b, :, 0] = (lo << np.uint64(1)) ^ carry
        table[b, :, 1] = (hi << np.uint64(1)) | (lo >> np.uint64(63))
    return table


# Flat `[16 * 256]` ghash, indexed `b * 256 + v` — one gather per medium
# position replaces that position's F128 multiply.
_CONVERT_DEV_G = ghash.to_ghash(fnp.asarray(_build_convert_table().reshape(-1, 2)))


def _fold_small(rows, *, bits: bool):
    """Fold the 3 small dims of byte `[n_rows, ell]` rows onto `[n_out, 16, ell]`,
    as `Σ_K xᴷ · rows[(o << 7) | (med << 3) | K]` in F8.

    `eq_small[K] = SG[0] · αᴷ` with α = φ₈(x), and φ₈ is a field homomorphism,
    so `Σ_K eq_small[K] · φ₈(y_K) = SG[0] · φ₈(Σ_K xᴷ · y_K)` — the 8 F128
    multiplies collapse into 8 u16 shift-XORs.

    `bits` says which track this is, and both halves of the answer follow from
    it. The AB track folds F8 products, whose shifted sum reaches 15 bits and
    needs the AES-poly reduce. The C track folds witness bits — and reads them
    the way the composite does, as `!= 0` rather than as bytes, because
    `_round1_input_rows` passes some input forms through uncoerced. That
    coercion is what bounds `Σ_K c_K · 2ᴷ` below 256, so the fold is a no-op
    and the convert-table gather below cannot run off the end of its row."""
    planes = rows.reshape(-1, _MEDIUM, _SMALL, rows.shape[-1])
    if bits:
        planes = planes != 0
    acc = planes[:, :, 0].astype(fnp.uint16)
    for k in range(1, _SMALL):
        acc = acc ^ (planes[:, :, k].astype(fnp.uint16) << k)
    return acc if bits else _gf8_reduce(acc)


def _fold_medium(values, eq_outer_scaled):
    """Fold the 4 medium dims and then the outer dims of a `[n_out, 16, ell]`
    byte plane onto one ghash `[ell]`.

    `eq_med[b] = MG[0] · γᵇ`, so `Σ_b eq_med[b] · φ₈(v_b)` is a row of the
    convert table — 16 gathers and 16 XORs where the naive form spends 16 F128
    multiplies. Only the outer fold that follows stays a GHASH multiply, one
    per (outer chunk, λ)."""
    offset = fnp.arange(_MEDIUM, dtype=fnp.int32)[None, :, None] * 256
    chunk = fnp.sum(_CONVERT_DEV_G[offset + values.astype(fnp.int32)], axis=1)
    return fnp.sum(chunk * eq_outer_scaled[:, None], axis=0)


def _extend_factor(x, n_rows: int, k_skip: int):
    """S→Λ extension of one round-1 factor -> AES rows `[n_rows, 2^k_skip]`.

    The packed F128 witness at k_skip = 6 goes through the circulant byte
    table: its flat uint64 view IS the table's 64-bit row, so that form skips
    both the NTT pair and the 8x bit expansion `_round1_input_rows` would
    materialize. Every other form — an unpacked bit array, raw F8 bytes the
    table's bit index cannot address, another k_skip — keeps the NTT."""
    if k_skip == _EXT_TABLE_K_SKIP and is_packed_witness(x):
        return lax.bitcast_convert_type(_extend_packed_rows(x.reshape(-1)), _AES)
    return _extend_rows(_round1_input_rows(x, n_rows), k_skip)


@functools.partial(frx.jit, static_argnums=(3,))
def _round1_core_factored(a, b, c, k_skip, r):
    """`_round1_core`'s message over the factored row-eq — the same (P^AB, P^C).

    `build_eq` pairs challenge i with row bit i, so with row =
    `(o << 7) | (b << 3) | K` the row-eq factors as `EO[o] · MG[b] · SG[K]`,
    and the two inner factors are geometric under the protocol's pinned inner
    challenges (`_inner_challenges_pinned` guards that). Folding them in that
    order — small by shift-reduce, medium by convert table, outer by one GHASH
    multiply — keeps the whole contraction on gf8 bytes until the last step,
    where the composite lifts every product bit to F128 first. `SG[0] · MG[0]`
    is the constant both folds drop; it rides into the outer eq table.

    Same formulation as `_urm_pallas`, in ordinary array primitives so it
    lowers on CPU."""
    outer_point = r[k_skip:]
    n_rows = 1 << outer_point.shape[0]
    sg = sumcheck.build_eq(outer_point[:_N_SMALL])
    mg = sumcheck.build_eq(outer_point[_N_SMALL : _N_SMALL + _N_MEDIUM])
    eo = sumcheck.build_eq(outer_point[_N_SMALL + _N_MEDIUM :]) * (sg[0] * mg[0])

    a_l, b_l = (_extend_factor(x, n_rows, k_skip) for x in (a, b))
    ab = _to_u8(a_l * b_l)
    p_ab = _fold_medium(_fold_small(ab, bits=False), eo)
    p_c = _fold_medium(_fold_small(_round1_input_rows(c, n_rows), bits=True), eo)
    return p_ab, _extend_folded_c(p_c, k_skip)


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


@functools.cache
def _pinned_inner_lanes() -> np.ndarray:
    """The protocol's pinned small+medium challenges as uint64 lanes [7, 2].

    Cached: `medium_challenges` runs host field inverses, and the dispatch
    guard consults this once per prove."""
    # Deferred: prover imports this module.
    from flock_zorch.zerocheck import prover

    return np.concatenate([prover.small_challenges(), prover.medium_challenges()])


def _inner_challenges_pinned(k_skip: int, r) -> bool:
    """Are `r`'s inner 7 coordinates the protocol's pinned small/medium
    challenges?

    Both factored formulations rest on `eq_small` and `eq_med` being geometric,
    which holds only for those values, so the guard compares them byte-wise — a
    7-scalar host pull, once per prove. Everything that fails here takes the
    composite, which is correct for any challenges, and a traced `r` (a caller
    that jits the whole prove rather than its stages) fails here for exactly
    that reason: its values are not readable at trace time."""
    expected = _pinned_inner_lanes()
    n_inner = expected.shape[0]
    if isinstance(r, frx.core.Tracer) or r.shape[0] < k_skip + n_inner:
        return False
    inner = np.asarray(ghash.to_lanes(r[k_skip : k_skip + n_inner]))
    return bool((inner == expected).all())


def _round1_pallas_ok(a, b, c, m: int, k_skip: int, r) -> bool:
    """May round-1 run on the Triton factored-eq kernel for this call?"""
    if frx.default_backend() != "gpu" or k_skip != 6:
        return False
    if not all(is_packed_witness(x) for x in (a, b, c)):
        return False
    # Deferred: _urm_pallas imports this module.
    from flock_zorch.zerocheck import _urm_pallas

    if (1 << (m - k_skip)) < _urm_pallas._ROWS_PER_PARTIAL:
        return False  # the grid is whole programs
    return _inner_challenges_pinned(k_skip, r)


def _round1_factored_ok(m: int, k_skip: int, r) -> bool:
    """May round-1 run on the portable factored-eq core for this call?

    CPU only: it is the tier that has no custom emitter, so the composite it
    replaces there lowers as the naive per-row form. On GPU the composite IS
    the fused emitter's operand and the Triton kernel already carries this
    formulation, so nothing would be gained by diverting either."""
    if frx.default_backend() != "cpu":
        return False
    if m - k_skip < _N_SMALL + _N_MEDIUM:
        return False  # no outer dims left once the inner 7 are folded
    return _inner_challenges_pinned(k_skip, r)


def round1_rows(a, b, c, m: int, k_skip: int, r):
    """Round-1 univariate-skip message (P^AB, P^C), each F128 [2^k_skip] on Λ,
    from device witness rows (uint8 [2^(m-k_skip), 2^k_skip]) — split from
    `witness_to_rows` so the witness is transferred once and reused by
    `zerocheck._fold_at_z`. Per row of 2^k_skip bits -> F8 col, inv-NTT on S then
    fwd-NTT on Λ, then accumulate eq(r[k_skip:], x) · φ₈(a·b) and · φ₈(c).
    Byte-identical to flock's `round1_naive` (== the wire `round1_ab`/`round1_c`).
    Returns (P^AB, P^C) as device-resident `binary_field_ghash [2^k_skip]` — no
    host lift; consumers observe/interpolate natively and byte-gate readers
    normalize via `ghash.to_lanes`.

    Under the pinned inner challenges the message is computed over the factored
    row-eq: on GPU with the packed F128 witness by the Triton kernel (1.7x the
    composite kernel at the 2^22-row block geometry, and it reads eq_out — 16 B
    per 128 rows — instead of a materialized row-eq), on CPU by
    `_round1_core_factored`. The composite stays the fallback for any other
    challenges and the byte oracle both are gated against, the same split
    `witness_blake3` uses."""
    if _round1_pallas_ok(a, b, c, m, k_skip, r):
        from flock_zorch.zerocheck import _urm_pallas  # deferred: import cycle

        return _urm_pallas.round1_core_pallas(a, b, c, k_skip, r)
    if _round1_factored_ok(m, k_skip, r):
        return _round1_core_factored(a, b, c, k_skip, r)
    return _round1_core(a, b, c, k_skip, r)  # eqx build + extend+phi+accum, fused
