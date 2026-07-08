"""Multilinear-sumcheck arithmetic core, authored in jax — byte-identical to
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

import jax
import jax.numpy as jnp

from flock_zorch import field

U64 = jnp.uint64
ONE = jnp.asarray([1, 0], dtype=U64)  # F128::ONE = {lo: 1, hi: 0}


# The uint64-lane <-> binary_field_ghash bitcast lives in `field` (one source of
# truth; the ghash->uint64 direction routes through uint32 lanes, so it is not a
# plain bitcast — see field.to_ghash).
_to_ghash = field.to_ghash
_from_ghash = field.from_ghash


_ONE_G = _to_ghash(ONE)  # scalar binary_field_ghash one


def build_eq(r):
    """eq evaluation table over `r`: `out[x] = ∏_i ((1+r_i)·(1⊕x_i) + r_i·x_i)`.

    r: uint64 [n, 2]; returns uint64 [2^n, 2]. flock builds this by power-of-two
    doubling (`univariate_skip::build_eq`): after absorbing r_i, bit i becomes the
    new high bit — the bit-0 half scales by (1+r_i), the bit-1 half by r_i. Each
    layer is one elementwise multiply over a doubling table → fully parallel per
    layer, n sequential layers (n static).
    """
    rg = _to_ghash(r)                    # [n]
    n = int(rg.shape[0])
    t = _ONE_G.reshape(1)                # [1]
    for i in range(n):
        r_i = rg[i]                      # scalar
        one_minus = r_i + _ONE_G         # (1 + r_i)
        t = jnp.concatenate([t * one_minus, t * r_i], axis=0)
    return _from_ghash(t)


_BUILD_EQ_FUSED = jax.jit(build_eq)


def build_eq_fused(r):
    """`build_eq` fused into ONE kernel. Byte-identical to `build_eq`, for eager
    call sites (round-1 URM, lincheck `eq_outer`) where the n doubling layers would
    otherwise dispatch eagerly with per-layer HBM materialization (~5 ms at n=20 vs
    ~0.2 ms fused). Inside an outer jit just call `build_eq` directly — it already
    fuses there."""
    return _BUILD_EQ_FUSED(jnp.asarray(r))


def fold_single(a, challenge):
    """Bind the low variable of one multilinear at `challenge` (flock
    `fold_in_place_single`): `out[x] = a[2x] + challenge·(a[2x+1] + a[2x])`.

    a: uint64 [2^k, 2] (k ≥ 1); returns uint64 [2^(k-1), 2].
    """
    ag = _to_ghash(a).reshape(-1, 2)
    a0, a1 = ag[:, 0], ag[:, 1]
    cg = _to_ghash(challenge)
    return _from_ghash(a0 + cg * (a0 + a1))


def fold_pair(a, b, challenge):
    """Bind the low variable of a pair (a, b) at `challenge` (flock
    `fold_in_place_pair`). Returns (a_folded, b_folded), each half-length.
    """
    return fold_single(a, challenge), fold_single(b, challenge)


def round_pair(a_mlv, b_mlv, r):
    """Multilinear-sumcheck round message for the AB pair (flock
    `round_pair_naive`). Returns `(r[0]·G(1), G(∞))`, each uint64 [2].

    a_mlv, b_mlv: uint64 [2^log_n, 2] (log_n ≥ 1); r: uint64 [log_n, 2], where
    r[0] is this round's bound-variable challenge and r[1:] is the eq over the
    remaining variables.
    """
    r = jnp.asarray(r, U64)
    eq = _to_ghash(build_eq(r[1:]))      # [2^(log_n-1)]
    r0 = _to_ghash(r[0])                  # scalar
    ag = _to_ghash(a_mlv).reshape(-1, 2)
    bg = _to_ghash(b_mlv).reshape(-1, 2)
    a0, a1 = ag[:, 0], ag[:, 1]
    b0, b1 = bg[:, 0], bg[:, 1]
    g_one = jnp.sum(eq * (a1 * b1))              # Σ eq·a1·b1
    g_inf = jnp.sum(eq * ((a0 + a1) * (b0 + b1)))  # Σ eq·(a0+a1)(b0+b1)
    return _from_ghash(r0 * g_one), _from_ghash(g_inf)


def eq_eval(r, x):
    """Point evaluation `eq(r, x) = ∏_i (1 + r_i + x_i)` (flock `eq_eval`).

    r, x: uint64 [n, 2]; returns uint64 [2]. Char-2 form of (1-r)(1-x) + r·x.
    """
    factors = _to_ghash(r) + _to_ghash(x) + _ONE_G  # [n]
    acc = _ONE_G
    for i in range(int(factors.shape[0])):
        acc = acc * factors[i]
    return _from_ghash(acc)
