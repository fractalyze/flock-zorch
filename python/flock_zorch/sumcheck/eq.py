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

import frx.numpy as jnp

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck.domain import compressed_domain, fold, summand_evals
from zorch.sumcheck.prover import ProductSummand

from flock_zorch import ghash

_PRODUCT2 = ProductSummand(2)._combine

U64 = jnp.uint64
ONE = jnp.asarray([1, 0], dtype=U64)  # F128::ONE = {lo: 1, hi: 0}
_ONE_G = ghash.to_ghash(ONE)  # scalar binary_field_ghash one


def build_eq(rg):
    """eq evaluation table over `rg`: `out[x] = ∏_i ((1+r_i)·(1⊕x_i) + r_i·x_i)`,
    `[n]` ghash challenges -> `[2^n]` ghash table, via zorch's
    `expand_eq_to_hypercube` (msb=True places r_i at bit i; its `(1−r_i)` share
    equals flock's `(1+r_i)` over char 2). flock builds this by power-of-two
    doubling (`univariate_skip::build_eq`): after absorbing r_i, bit i becomes the
    new high bit — one elementwise multiply per layer, n sequential layers."""
    return expand_eq_to_hypercube(rg, _ONE_G, msb=True)


def build_eq_lanes(r):
    """`build_eq` on the uint64 [n, 2] lane layout -> uint64 [2^n, 2] — the public
    golden/test I/O contract; the compute is ghash, bridged at the boundary."""
    return ghash.from_ghash(build_eq(ghash.to_ghash(r)))


def fold_single(a, challenge):
    """Bind the low variable of one multilinear at `challenge` (flock
    `fold_in_place_single`): `out[x] = a[2x] + challenge·(a[2x+1] + a[2x])` — zorch's
    LSB `fold` (`split_pairs`; over char 2 `(p0+p1) == (p1−p0)`).

    a: uint64 [2^k, 2] (k ≥ 1); returns uint64 [2^(k-1), 2].
    """
    return ghash.from_ghash(
        fold(ghash.to_ghash(a), ghash.to_ghash(challenge), msb=False))


def fold_pair(a, b, challenge):
    """Bind the low variable of a pair (a, b) at `challenge` (flock
    `fold_in_place_pair`). Returns (a_folded, b_folded), each half-length.
    """
    return fold_single(a, challenge), fold_single(b, challenge)


def build_eq_suffix_tables(cs_g):
    """eq tables for every challenge suffix: absorbing `cs_g[i]` as the low bit
    of `eq(cs_g[i+1:])` yields `eq(cs_g[i:])`, so all n+1 tables cost one
    doubling chain — n mul layers — instead of n separate `build_eq` builds
    (each layer is a fat clmul kernel XLA compiles for ~0.7 s, so a per-round
    rebuild multiplied that by the round count). Values match per-suffix
    `build_eq` exactly: GF mul is exact, associative, commutative.

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


def round_pair_eq(ag, bg, eq, r0g):
    """`round_pair` with the eq table already built — the per-round core for
    callers that precompute every suffix table once (`build_eq_suffix_tables`).

    The message `[G(1), G(∞)]` is zorch's compressed product round on the low bind:
    `summand_evals` over `compressed_domain(1)` with the eq suffix as the per-point
    weight and `msb=False` (`s(∞)`'s char-2 `(a1−a0)` is flock's `(a0+a1)`)."""
    g_one, g_inf = summand_evals(
        jnp.stack([ag, bg]), _PRODUCT2, compressed_domain(1, ag.dtype),
        weight=eq, msb=False)
    return r0g * g_one, g_inf


def round_pair(ag, bg, rg):
    """Multilinear-sumcheck round message for the AB pair, native ghash (flock
    `round_pair_naive`): `[2^log_n]` factor pair + `[log_n]` challenges -> the
    scalar message pair `(r[0]·G(1), G(∞))`. rg[0] is this round's bound-variable
    challenge, rg[1:] the eq over the remaining variables."""
    return round_pair_eq(ag, bg, build_eq(rg[1:]), rg[0])


def round_pair_lanes(a_mlv, b_mlv, r):
    """`round_pair` on the uint64 [.., 2] lane layout — the public golden/test I/O
    contract; returns `(r[0]·G(1), G(∞))`, each uint64 [2]."""
    g1, ginf = round_pair(ghash.to_ghash(a_mlv), ghash.to_ghash(b_mlv),
                          ghash.to_ghash(jnp.asarray(r, U64)))
    return ghash.from_ghash(g1), ghash.from_ghash(ginf)
