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

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck.domain import compressed_domain, summand_evals
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
        t = fnp.stack([t * (c + _ONE_G), t * c], axis=1).reshape(-1)
        out.append(t)
    return out[::-1]


def round_pair_eq(ag, bg, eq, r0g):
    """The per-round message core, taking an eq table the caller precomputed
    (one `build_eq_suffix_tables` chain serves every round).

    The message `[G(1), G(∞)]` is zorch's compressed product round on the low bind:
    `summand_evals` over `compressed_domain(1)` with the eq suffix as the per-point
    weight and `msb=False` (`s(∞)`'s char-2 `(a1−a0)` is flock's `(a0+a1)`)."""
    g_one, g_inf = summand_evals(
        fnp.stack([ag, bg]), _PRODUCT2, compressed_domain(1, ag.dtype),
        weight=eq, msb=False)
    return r0g * g_one, g_inf
