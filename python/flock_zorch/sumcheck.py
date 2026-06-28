"""Multilinear-sumcheck arithmetic core, authored in jax — byte-identical to
flock's `zerocheck::{univariate_skip,multilinear}` primitives.

These are the reusable kernels shared by BOTH sumchecks in flock's PIOP
(zerocheck and lincheck): the eq-table expansion, the multilinear fold, and the
per-round prover message. They are pure GF(2^128) arithmetic over uint64 lanes,
so they inherit the clmad FFI on GPU (pass `mul=field_clmad.mul`) and are fully
data-parallel — the multilinear sumcheck is the prover's biggest GPU win.

Conventions match flock exactly:
  * Field add is XOR; `1 + r` is `r ^ ONE` (ONE = [1, 0] in [lo, hi] lanes).
  * The LOW bit of a multilinear index is bound first: the pair
    (f[2x], f[2x+1]) is (X=0, X=1). `build_eq(r)` places r_i at bit i.
  * `round_pair` sends `(r[0]·G(1), G(∞))` — flock's Karatsuba ∞-trick message,
    where G(X) = Σ_x' eq(r[1:], x')·a(X,x')·b(X,x') and the wire polynomial is
    Π(X) = eq(r[0], X)·G(X) (so Π(1) = r[0]·G(1), leading coeff G(∞)).

Requires `jax_enable_x64`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from flock_zorch import field

U64 = jnp.uint64
ONE = jnp.asarray([1, 0], dtype=U64)  # F128::ONE = {lo: 1, hi: 0}


def build_eq(r, mul=field.mul):
    """eq evaluation table over `r`: `out[x] = ∏_i ((1+r_i)·(1⊕x_i) + r_i·x_i)`.

    r: uint64 [n, 2]; returns uint64 [2^n, 2]. flock builds this by power-of-two
    doubling (`univariate_skip::build_eq`): after absorbing r_i, bit i becomes the
    new high bit — the bit-0 half scales by (1+r_i), the bit-1 half by r_i. Each
    layer is one elementwise multiply over a doubling table → fully parallel per
    layer, n sequential layers (n static).
    """
    n = int(r.shape[0])
    t = ONE.reshape(1, 2)  # [1, 2] = {ONE}
    for i in range(n):
        r_i = r[i]             # [2]
        one_minus = r_i ^ ONE  # (1 + r_i)
        t = jnp.concatenate([mul(t, one_minus), mul(t, r_i)], axis=0)
    return t


_BUILD_EQ_CACHE: dict = {}


def build_eq_fused(r, mul=field.mul):
    """`build_eq` fused into ONE kernel (cached per `mul`). Byte-identical to
    `build_eq`, for eager call sites (round-1 URM, lincheck `eq_outer`) where the
    n doubling layers would otherwise dispatch eagerly with per-layer HBM
    materialization (~5 ms at n=20 vs ~0.2 ms fused). Inside an outer jit just call
    `build_eq` directly — it already fuses there."""
    fn = _BUILD_EQ_CACHE.get(mul)
    if fn is None:
        fn = jax.jit(lambda rr: build_eq(rr, mul=mul))
        _BUILD_EQ_CACHE[mul] = fn
    return fn(jnp.asarray(r))


def fold_single(a, challenge, mul=field.mul):
    """Bind the low variable of one multilinear at `challenge` (flock
    `fold_in_place_single`): `out[x] = a[2x] + challenge·(a[2x+1] + a[2x])`.

    a: uint64 [2^k, 2] (k ≥ 1); returns uint64 [2^(k-1), 2].
    """
    ap = a.reshape(-1, 2, 2)
    a0, a1 = ap[:, 0, :], ap[:, 1, :]
    return a0 ^ mul(challenge, a0 ^ a1)


def fold_pair(a, b, challenge, mul=field.mul):
    """Bind the low variable of a pair (a, b) at `challenge` (flock
    `fold_in_place_pair`). Returns (a_folded, b_folded), each half-length.
    """
    return fold_single(a, challenge, mul), fold_single(b, challenge, mul)


def round_pair(a_mlv, b_mlv, r, mul=field.mul):
    """Multilinear-sumcheck round message for the AB pair (flock
    `round_pair_naive`). Returns `(r[0]·G(1), G(∞))`, each uint64 [2].

    a_mlv, b_mlv: uint64 [2^log_n, 2] (log_n ≥ 1); r: uint64 [log_n, 2], where
    r[0] is this round's bound-variable challenge and r[1:] is the eq over the
    remaining variables.
    """
    eq = build_eq(r[1:], mul=mul)             # [2^(log_n-1), 2]
    ap = a_mlv.reshape(-1, 2, 2)
    bp = b_mlv.reshape(-1, 2, 2)
    a0, a1 = ap[:, 0, :], ap[:, 1, :]
    b0, b1 = bp[:, 0, :], bp[:, 1, :]
    g_one = field.sum(mul(eq, mul(a1, b1)))             # Σ eq·a1·b1
    g_inf = field.sum(mul(eq, mul(a0 ^ a1, b0 ^ b1)))   # Σ eq·(a0+a1)(b0+b1)
    return mul(r[0], g_one), g_inf


def eq_eval(r, x, mul=field.mul):
    """Point evaluation `eq(r, x) = ∏_i (1 + r_i + x_i)` (flock `eq_eval`).

    r, x: uint64 [n, 2]; returns uint64 [2]. Char-2 form of (1-r)(1-x) + r·x.
    """
    factors = r ^ x ^ ONE  # [n, 2]
    acc = ONE
    for i in range(int(r.shape[0])):
        acc = mul(acc, factors[i])
    return acc
