"""Multilinear-sumcheck arithmetic core, authored in frx — byte-identical to
flock's `zerocheck::{univariate_skip,multilinear}` primitives.

These are the reusable kernels shared by BOTH sumchecks in flock's PIOP
(zerocheck and lincheck): the eq-table expansion, the multilinear fold, and the
per-round prover message. The GF(2^128) arithmetic runs on the native
`binary_field_ghash` dtype (multiply → `*`, add → `+`, sum → `jnp.sum`), so it
is fully data-parallel and uses the dtype's hardware-CLMUL multiply on GPU — the
multilinear sumcheck is the prover's biggest GPU win.

The public functions keep flock's `uint64 [..., 2] = [lo, hi]` I/O contract (the
golden gates and callers pass/read that layout); the dtype is used only for the
internal compute, bridged at the boundary.

Conventions match flock exactly:
  * Field add is XOR; `1 + r` is `r + ONE`.
  * The LOW bit of a multilinear index is bound first: the pair
    (f[2x], f[2x+1]) is (X=0, X=1). `build_eq(r)` places r_i at bit i.
  * `round_pair` sends `(r[0]·G(1), G(∞))` — flock's Karatsuba ∞-trick message,
    where G(X) = Σ_x' eq(r[1:], x')·a(X,x')·b(X,x') and the wire polynomial is
    Π(X) = eq(r[0], X)·G(X) (so Π(1) = r[0]·G(1), leading coeff G(∞)).

Requires `jax_enable_x64`.
"""
from __future__ import annotations

import frx
import frx.numpy as jnp

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck.domain import compressed_domain, fold, summand_evals
from zorch.sumcheck.prover import ProductSummand

from flock_zorch import field

_PRODUCT2 = ProductSummand(2)._combine

U64 = jnp.uint64
ONE = jnp.asarray([1, 0], dtype=U64)  # F128::ONE = {lo: 1, hi: 0}
_ONE_G = field.to_ghash(ONE)  # scalar binary_field_ghash one


def build_eq_g(rg):
    """`build_eq` on native ghash: `[n]` challenges -> `[2^n]` eq table, via zorch's
    `expand_eq_to_hypercube` (msb=True places r_i at bit i; its `(1−r_i)` share
    equals flock's `(1+r_i)` over char 2). Ghash-native so a jitted caller keeps its
    whole trace on the dtype, with no in-trace lane bitcasts to fuse around."""
    return expand_eq_to_hypercube(rg, _ONE_G, msb=True)


def build_eq(r):
    """eq evaluation table over `r`: `out[x] = ∏_i ((1+r_i)·(1⊕x_i) + r_i·x_i)`.

    r: uint64 [n, 2]; returns uint64 [2^n, 2]. flock builds this by power-of-two
    doubling (`univariate_skip::build_eq`): after absorbing r_i, bit i becomes the
    new high bit — the bit-0 half scales by (1+r_i), the bit-1 half by r_i. Each
    layer is one elementwise multiply over a doubling table → fully parallel per
    layer, n sequential layers (n static).
    """
    return field.from_ghash(build_eq_g(field.to_ghash(r)))


_BUILD_EQ_FUSED = frx.jit(build_eq)


def build_eq_fused(r):
    """`build_eq` fused into ONE kernel. Byte-identical to `build_eq`, for eager
    call sites (round-1 URM, lincheck `eq_outer`) where the n doubling layers would
    otherwise dispatch eagerly with per-layer HBM materialization (~5 ms at n=20 vs
    ~0.2 ms fused). Inside an outer jit just call `build_eq` directly — it already
    fuses there."""
    return _BUILD_EQ_FUSED(jnp.asarray(r))


_BUILD_EQ_FUSED_G = frx.jit(lambda r: build_eq_g(field.to_ghash(r)))


def build_eq_fused_g(r):
    """`build_eq_fused` returning native ghash — for ghash-consuming callers that
    would otherwise bitcast the uint64 result straight back with `to_ghash`."""
    return _BUILD_EQ_FUSED_G(jnp.asarray(r, U64))


def fold_single(a, challenge):
    """Bind the low variable of one multilinear at `challenge` (flock
    `fold_in_place_single`): `out[x] = a[2x] + challenge·(a[2x+1] + a[2x])` — zorch's
    LSB `fold` (`split_pairs`; over char 2 `(p0+p1) == (p1−p0)`).

    a: uint64 [2^k, 2] (k ≥ 1); returns uint64 [2^(k-1), 2].
    """
    return field.from_ghash(
        fold(field.to_ghash(a), field.to_ghash(challenge), msb=False))


def build_eq_suffix_tables_g(cs_g):
    """eq tables for every challenge suffix: absorbing `cs_g[i]` as the low bit
    of `eq(cs_g[i+1:])` yields `eq(cs_g[i:])`, so all n+1 tables cost one
    doubling chain — n mul layers — instead of n separate `build_eq_g` builds
    (each layer is a fat clmul kernel XLA compiles for ~0.7 s, so a per-round
    rebuild multiplied that by the round count). Values match per-suffix
    `build_eq_g` exactly: GF mul is exact, associative, commutative.

    cs_g: `[n]` ghash challenges. Returns `[T_0 .. T_n]`, `T_i = eq(cs_g[i:])`
    of shape `[2^(n-i)]`; `T_n = [1]`."""
    n = int(cs_g.shape[0])
    t = _ONE_G.reshape(1)
    out = [t]
    for i in range(n - 1, -1, -1):
        c = cs_g[i]
        t = jnp.stack([t * (c + _ONE_G), t * c], axis=1).reshape(-1)
        out.append(t)
    return out[::-1]


def round_pair_eq_g(ag, bg, eq, r0g):
    """`round_pair_g` with the eq table already built — the per-round core for
    callers that precompute every suffix table once (`build_eq_suffix_tables_g`).

    The message `[G(1), G(∞)]` is zorch's compressed product round on the low bind:
    `summand_evals` over `compressed_domain(1)` with the eq suffix as the per-point
    weight and `msb=False` (`s(∞)`'s char-2 `(a1−a0)` is flock's `(a0+a1)`)."""
    g_one, g_inf = summand_evals(
        jnp.stack([ag, bg]), _PRODUCT2, compressed_domain(1, ag.dtype),
        weight=eq, msb=False)
    return r0g * g_one, g_inf


def round_pair_g(ag, bg, rg):
    """`round_pair` on native ghash: `[2^log_n]` factor pair + `[log_n]`
    challenges -> the scalar message pair `(r[0]·G(1), G(∞))`."""
    return round_pair_eq_g(ag, bg, build_eq_g(rg[1:]), rg[0])


def round_pair(a_mlv, b_mlv, r):
    """Multilinear-sumcheck round message for the AB pair (flock
    `round_pair_naive`). Returns `(r[0]·G(1), G(∞))`, each uint64 [2].

    a_mlv, b_mlv: uint64 [2^log_n, 2] (log_n ≥ 1); r: uint64 [log_n, 2], where
    r[0] is this round's bound-variable challenge and r[1:] is the eq over the
    remaining variables.
    """
    g1, ginf = round_pair_g(field.to_ghash(a_mlv), field.to_ghash(b_mlv),
                            field.to_ghash(jnp.asarray(r, U64)))
    return field.from_ghash(g1), field.from_ghash(ginf)
