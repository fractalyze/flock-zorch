"""Multilinear-sumcheck arithmetic core, authored in frx — byte-identical to
flock's `zerocheck::{univariate_skip,multilinear}` primitives.

These are the reusable kernels shared by BOTH sumchecks in flock's PIOP
(zerocheck and lincheck): the eq-table expansion, the multilinear fold, and the
per-round prover message. The GF(2^128) arithmetic runs on the native
`binary_field_ghash` dtype (multiply → `*`, add → `+`, sum → `fnp.sum`), so it
is fully data-parallel and uses the dtype's hardware-CLMUL multiply on GPU — the
multilinear sumcheck is the prover's biggest GPU win.

Everything runs on the native dtype end to end; the proof messages leave it
only where they are serialized.

Conventions match flock exactly:
  * Field add is XOR; `1 + r` is `r + ONE`.
  * The LOW bit of a multilinear index is bound first: the pair
    (f[2x], f[2x+1]) is (X=0, X=1). `build_eq(r)` places r_i at bit i.
  * `round_pair_eq` sends `(r[0]·G(1), G(∞))` — flock's Karatsuba ∞-trick
    message, where G(X) = Σ_x' eq(r[1:], x')·a(X,x')·b(X,x') and the wire
    polynomial is Π(X) = eq(r[0], X)·G(X) (so Π(1) = r[0]·G(1), leading
    coeff G(∞)).

Requires `jax_enable_x64`.
"""

from __future__ import annotations

import frx.numpy as fnp
from zorch.poly.eq import expand_eq_to_hypercube, expand_hypercube_step
from zorch.sumcheck.domain import compressed_domain, split_pairs, summand_evals
from zorch.sumcheck.prover import ProductSummand

from flock_zorch import ghash

_PRODUCT2 = ProductSummand(2)._combine

U64 = fnp.uint64
ONE = fnp.asarray([1, 0], dtype=U64)  # F128::ONE = {lo: 1, hi: 0}
_ONE_G = ghash.to_ghash(ONE)  # scalar binary_field_ghash one


def build_eq(rg):
    """eq evaluation table over `rg`: `out[x] = ∏_i ((1+r_i)·(1⊕x_i) + r_i·x_i)`,
    `[n]` ghash challenges -> `[2^n]` ghash table, via zorch's
    `expand_eq_to_hypercube` (msb=True places r_i at bit i; its `(1−r_i)` share
    equals flock's `(1+r_i)` over char 2). flock builds this by power-of-two
    doubling (`univariate_skip::build_eq`): after absorbing r_i, bit i becomes the
    new high bit — one elementwise multiply per layer, n sequential layers."""
    return expand_eq_to_hypercube(rg, _ONE_G, msb=True)


# Above this suffix length, tables are emitted as outer products of two half
# tables instead of extending one doubling chain. XLA fuses a chain's tail
# layers into interleave fusions that recompute each output element's ancestor
# factors (zorch#609's finding for the concatenate form of the same chain), so
# every large layer pays several GF(2^128) muls per element plus a full
# read-back of its predecessor; an outer product is one mul per element over
# two small inputs — a pure streaming write. Same rationale as zorch's
# `poly.eq._OUTER_SPLIT_MIN` but an independent knob: this one gates a family
# emission whose high half is amortized across every larger suffix, so the
# crossovers need not track each other.
_OUTER_SPLIT_MIN = 16


def _suffix_chain(cs_g):
    """The doubling chain: absorbing `cs_g[i]` as the low bit of
    `eq(cs_g[i+1:])` yields `eq(cs_g[i:])`, one interleave layer per
    challenge. Returns `[T_0 .. T_n]` like `build_eq_suffix_tables`."""
    t = _ONE_G.reshape(1)
    out = [t]
    for i in range(int(cs_g.shape[0]) - 1, -1, -1):
        t = expand_hypercube_step(t, cs_g[i], msb=False)
        out.append(t)
    return out[::-1]


def build_eq_suffix_tables(cs_g, keep=None):
    """eq tables for every challenge suffix, sharing work across the family:
    the small layers are shared chains instead of n separate `build_eq` builds
    (each layer is a fat clmul kernel XLA compiles for ~0.7 s, so a per-round
    rebuild multiplied that by the round count), and every table past
    `_OUTER_SPLIT_MIN` is emitted as ONE outer product: with a split point j,
    `T_i = T_j ⊗ eq(cs_g[i:j])` for every `i < j` — the high half `T_j` is one
    table shared by all larger suffixes, and the low factors `eq(cs_g[i:j])`
    are the suffix family of `cs_g[:j]`, so both sides recurse on half-size
    chains. Values match per-suffix `build_eq` exactly: GF mul is exact,
    associative, commutative.

    The dual build — one `build_eq(cs_g)` then `contract_hypercube_step` down —
    spends fewer muls (contraction is XOR-only) but re-reads each large
    predecessor, ~1.5x the traffic of these write-only emissions; the ML window
    is bandwidth-bound, so traffic is the currency.

    cs_g: `[n]` ghash challenges. Returns `[T_0 .. T_n]`, `T_i = eq(cs_g[i:])`
    of shape `[2^(n-i)]`; `T_n = [1]`.

    keep: optional predicate on the suffix index — the indices the caller
    will read. Kept slots are always present and exact; un-kept slots come
    back as None, and past the outer-product split their emissions are
    skipped entirely (below it the chain computes every layer anyway and XLA
    drops what nothing consumes). Why a consumer can afford to drop tables
    is the consumer's story — e.g. `round_pair_eq_sq_basis` for the sq pair
    schedule."""
    tables = _suffix_family(cs_g, (lambda i: True) if keep is None else keep)
    if keep is None:
        return tables
    return [t if keep(i) else None for i, t in enumerate(tables)]


def _suffix_family(cs_g, keep):
    """The recursion behind `build_eq_suffix_tables`. May return un-kept
    entries that exist as building blocks anyway (chain layers, shared high
    halves) — the public wrapper masks those to None, so the keep contract
    stays predicate-defined rather than split-regime-defined."""
    n = int(cs_g.shape[0])
    if n < _OUTER_SPLIT_MIN:
        return _suffix_chain(cs_g)
    j = n // 2
    low = _suffix_family(cs_g[:j], keep)  # low[i] = eq(cs_g[i:j])
    # Of the high family the parent needs the shared high half T_j (local 0)
    # as a building block besides the kept globals.
    high = _suffix_family(cs_g[j:], lambda i: i == 0 or keep(j + i))
    t_j = high[0]
    # T_i places cs_g[i] at bit 0, so bits [j-i, n-i) of T_i are exactly T_j's
    # index — T_j is the slow axis, the low factor the fast axis.
    return [
        (t_j[:, None] * low[i][None, :]).reshape(-1) if keep(i) else None
        for i in range(j)
    ] + high


def round_pair_eq(ag, bg, eq, r0g):
    """The per-round message core, taking an eq table the caller precomputed
    (one `build_eq_suffix_tables` chain serves every round).

    The message `[G(1), G(∞)]` is zorch's compressed product round on the low bind:
    `summand_evals` over `compressed_domain(1)` with the eq suffix as the per-point
    weight and `msb=False` (`s(∞)`'s char-2 `(a1−a0)` is flock's `(a0+a1)`)."""
    g_one, g_inf = summand_evals(
        fnp.stack([ag, bg]),
        _PRODUCT2,
        compressed_domain(1, ag.dtype),
        weight=eq,
        msb=False,
    )
    return r0g * g_one, g_inf


def sqrt_ghash(x):
    """√x in GF(2¹²⁸): x^(2¹²⁷), the inverse of the Frobenius x ↦ x², as 127
    squarings. Unique and total (x^(2¹²⁸) = x for every x, zero included), and an
    automorphism over char 2 — √ distributes over both `+` and `·`, which is what
    lets `build_eq_suffix_tables(sqrt_ghash(cs))` produce exactly √(eq table):
    every chain layer multiplies by √c or √c + 1 = √(c + 1)."""
    for _ in range(127):
        x = x * x
    return x


def round_pair_eq_sq(ag, eq_sqrt, r0g):
    """`round_pair_eq` for the equal-factor round (b ≡ a — the identity R1CS
    instance proves ẑ∘ẑ ⊕ ẑ = 0, so both product factors are the same
    multilinear; hash circuits keep distinct Az/Bz tracks and the generic
    round), at half the multiplies: every product is then a square, and
    char-2 squaring distributes over the XOR-sum, so

        Σ eq·a₁·a₁ = (Σ √eq·a₁)²   and   Σ eq·S·S = (Σ √eq·S)²,  S = a₀+a₁.

    Takes the √eq suffix table (`build_eq_suffix_tables` over `sqrt_ghash`
    challenges) in place of eq: one weighting mul per pair replaces the generic
    path's product + weighting pair — 2 muls/pair against 4 — and the closing
    squares are two scalar muls. The wire pair `[r₀·G(1), G(∞)]` is byte-identical:
    the rearrangement is an exact field identity, GF sums reorder freely.

    Equal factors are the identity's whole domain: char 2 has no polarization
    (x+y squared is x² + y²), so a distinct-factor product a₁·b₁ cannot be
    recovered from squares — this is an instance specialization, not a general
    round speedup."""
    a0, a1 = split_pairs(ag)
    g_one = fnp.sum(eq_sqrt * a1)
    g_inf = fnp.sum(eq_sqrt * (a0 + a1))
    return r0g * (g_one * g_one), g_inf * g_inf


def round_pair_eq_deferred(ag, bg, eq_next):
    """Round i+1's message precursors off round i's array read — the cascade
    composition's deferred half.

    Over groups of four (v0..v3 = ag[4z..4z+4]), round i's fold at the
    not-yet-sampled ρ_i sends the `[2z+1]` element to `v2 + ρ_i(v2+v3)` and the
    pair sum to `e + ρ_i(e+o)` (e = v0+v2, o = v1+v3), so round i+1's G(1) and
    G(∞) are quadratics in ρ_i whose Karatsuba coefficient triples
    `(w_lo, w_hi, w_sum)` are the six eq-weighted aggregates below — no second
    array read. Evaluate with `eval_deferred` once ρ_i is drawn; the caller
    scales G(1) by its r₀ as usual.

    The six reductions are hand-written rather than routed through
    `summand_evals` (cf. `round_pair_eq`): the operands stay strided views of
    the one array read and the Karatsuba structure stays legible — the same
    trade `round_pair_eq_sq` makes."""
    a4 = ag.reshape(-1, 4)
    b4 = bg.reshape(-1, 4)

    def _triple(x0, x1, y0, y1):
        return (
            fnp.sum(eq_next * (x0 * y0)),
            fnp.sum(eq_next * ((x0 + x1) * (y0 + y1))),
            fnp.sum(eq_next * (x1 * y1)),
        )

    g_one = _triple(a4[:, 2], a4[:, 3], b4[:, 2], b4[:, 3])
    e_a, o_a = a4[:, 0] + a4[:, 2], a4[:, 1] + a4[:, 3]
    e_b, o_b = b4[:, 0] + b4[:, 2], b4[:, 1] + b4[:, 3]
    g_inf = _triple(e_a, o_a, e_b, o_b)
    return g_one, g_inf


def eval_deferred(triple, rho):
    """Evaluate a `round_pair_eq_deferred` Karatsuba triple at the sampled ρ:
    the char-2 quadratic `w_lo + (w_lo+w_hi+w_sum)·ρ + w_hi·ρ²`."""
    lo, hi, s = triple
    return lo + (lo + hi + s) * rho + hi * (rho * rho)


def round_pair_eq_deferred_sq(ag, eq_sqrt_next):
    """`round_pair_eq_deferred` on the equal-factor path (b ≡ a).

    The sq round's message values are squares of √eq-weighted sums
    (`round_pair_eq_sq`), and round i's fold sends those sums to values LINEAR
    in the not-yet-sampled ρ_i — so char-2 squaring kills the cross term:
    `(A + ρB)² = A² + ρ²·B²`. Each deferred message value therefore needs a
    coefficient PAIR of √eq_next-weighted sums (one weighting mul each), not
    the generic path's Karatsuba triple. Over groups of four
    (v0..v3 = ag[4z..4z+4], e = v0+v2, o = v1+v3):

        G(1):  A = Σ √eq_next·v2,  B = Σ √eq_next·(v2+v3)
        G(∞):  C = Σ √eq_next·e,   D = Σ √eq_next·(e+o)

    Evaluate with `eval_deferred_sq` once ρ_i is drawn; the caller scales
    G(1) by its r₀ as usual.

    Superseded on the pair schedule by `round_pair_eq_sq_basis`, which
    derives the same four values from its shared base sums; retained as the
    independently-computed exactness oracle (`RoundPairEqSqBasisTest`) — do
    NOT re-express this over those sums, or the test stops testing."""
    a4 = ag.reshape(-1, 4)
    g_one = (
        fnp.sum(eq_sqrt_next * a4[:, 2]),
        fnp.sum(eq_sqrt_next * (a4[:, 2] + a4[:, 3])),
    )
    e, o = a4[:, 0] + a4[:, 2], a4[:, 1] + a4[:, 3]
    g_inf = (fnp.sum(eq_sqrt_next * e), fnp.sum(eq_sqrt_next * (e + o)))
    return g_one, g_inf


def eval_deferred_sq(pair, rho):
    """Evaluate a deferred sq coefficient pair (`round_pair_eq_sq_basis` on
    the production path, `round_pair_eq_deferred_sq` as its oracle) at the
    sampled ρ: `(A + ρB)² = A² + ρ²·B²` — char-2 squaring distributes, no
    cross term."""
    a, b = pair
    return a * a + (rho * rho) * (b * b)


def round_pair_eq_sq_basis(ag, eq_sqrt_next, sqrt_c, r0g):
    """The whole equal-factor pair — round i's message AND the deferred
    coefficient pairs — from FOUR base sums over the SMALLER table only.

    The √eq suffix chain doubles as `√T_i[2z] = (1+√c_i)·√T_{i+1}[z]`,
    `√T_i[2z+1] = √c_i·√T_{i+1}[z]` (√ distributes over + and · in char 2), so
    round i's √T_i-weighted sums split by z-parity into √T_{i+1}-weighted sums
    of the same groups of four the deferred coefficients already read. With
    `s_j = Σ √T_{i+1}·v_j` (v0..v3 = ag[4z..4z+4]), all six pair scalars are
    linear in s0..s3:

        G(1)_i  = (1+√c_i)·s1 + √c_i·s3          A = s2
        G(∞)_i  = (1+√c_i)·(s0+s1) + √c_i·(s2+s3) B = s2+s3
                                                  C = s0+s2
                                                  D = s0+s1+s2+s3

    against `round_pair_eq_sq(ag, √T_i)` + `round_pair_eq_deferred_sq(ag,
    √T_{i+1})`. Exact GF reassociation — same bytes — at half the weighting
    muls (four n/4 sums against the split forms' 2n) and with round i's table
    unread: on the pair schedule the even-index √eq tables (the largest
    included) go dead entirely — ~2/3 of the family's element count — so
    their emissions are skipped at the source
    (`build_eq_suffix_tables(keep=...)`).

    Equal-factor-specific: the generic pair's two bases share only the w_hi
    product, so re-basing there buys ~nothing — this is the sq squaring
    (messages are squares of LINEAR functionals of the array) paying out a
    second time.

    Returns `(m1, minf, g_one, g_inf)`: the wire message pair (G(1) already
    scaled by r₀ and squared) plus the two `eval_deferred_sq` coefficient
    pairs."""
    a4 = ag.reshape(-1, 4)
    s0 = fnp.sum(eq_sqrt_next * a4[:, 0])
    s1 = fnp.sum(eq_sqrt_next * a4[:, 1])
    s2 = fnp.sum(eq_sqrt_next * a4[:, 2])
    s3 = fnp.sum(eq_sqrt_next * a4[:, 3])
    w = sqrt_c + _ONE_G
    g_one = w * s1 + sqrt_c * s3
    g_inf = w * (s0 + s1) + sqrt_c * (s2 + s3)
    return (
        r0g * (g_one * g_one),
        g_inf * g_inf,
        (s2, s2 + s3),
        (s0 + s2, s0 + s1 + s2 + s3),
    )
