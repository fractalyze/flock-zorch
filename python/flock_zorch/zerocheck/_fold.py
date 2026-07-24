"""Device precompute for zerocheck's round-1 c-claim and round-2 binding — the
Lagrange-weight / batched-inverse / fold-at-z cluster, kept out of zerocheck.py so
prove_packed reads as the PIOP. Byte-identical to flock's zerocheck/{univariate_skip,
multilinear}. The deeply-sequential Lagrange/Fermat work runs on the native binary_field_ghash multiply.

Requires jax_enable_x64.
"""
from __future__ import annotations

import numpy as np
import frx
import frx.numpy as fnp

from flock_zorch import ghash
from flock_zorch.zerocheck import _urm

_ONE = np.array([1, 0], dtype=np.uint64)
_ONE_G = ghash.to_ghash(fnp.asarray(_ONE))  # binary_field_ghash scalar one


def _prod_axis1(mat):
    """F128 product over axis 1 of ghash [n, k] via log2(k) pairwise mul steps."""
    n = mat.shape[1]
    while n > 1:
        h = n // 2
        prod = mat[:, :h] * mat[:, h:2 * h]
        if n % 2:
            prod = fnp.concatenate([prod, mat[:, 2 * h:]], axis=1)
        mat = prod
        n = mat.shape[1]
    return mat[:, 0]


@frx.jit
def _lag_numden(sg, zg):
    """num[i]=Π_{j≠i}(z+s_j), den[i]=Π_{j≠i}(s_i+s_j); diagonal terms set to 1."""
    ell = sg.shape[0]
    eye = fnp.eye(ell, dtype=bool)
    num_mat = fnp.where(eye, _ONE_G, fnp.broadcast_to((zg + sg)[None, :], (ell, ell)))
    den_mat = fnp.where(eye, _ONE_G, sg[:, None] + sg[None, :])
    return _prod_axis1(num_mat), _prod_axis1(den_mat)


@frx.jit
def _lag_w(num, inv_den):
    return num * inv_den


@frx.jit
def _batch_inv(ag):
    """Batched GF(2^128) inverse a^(2^128-2) = Π_{k=1}^{127} a^(2^k), via 127
    square-and-multiply steps. Rolled into a `fori_loop`: the native ghash multiply
    unrolled 127x compiles pathologically slowly on the CPU backend (minutes),
    whereas one rolled body is O(1) to compile."""
    def body(_, carry):
        sq, result = carry
        sq = sq * sq
        return sq, result * sq

    _, result = frx.lax.fori_loop(0, 127, body, (ag, fnp.broadcast_to(_ONE_G, ag.shape)))
    return result


def _lagrange_weights(k_skip: int, zg, offset: int):
    """L_i(z) over the φ₈-embedded nodes PHI_8_TABLE[offset+i], i∈[0, 2^k_skip).
    offset=0 → the S domain; offset=2^k_skip → the Λ domain.

    Vectorized + jitted (replaces the scalar O(ell²) host-Python F128 double-loop;
    jit is essential — it keeps the native ghash multiplies fused).
    Same field math → byte-identical weights (gated).

    zg: `binary_field_ghash` scalar (z is a value in the L_i formula, not an index).
    Returns `binary_field_ghash [2^k_skip]` — never leaves the dtype."""
    sg = ghash.to_ghash(fnp.asarray(_urm.PHI_8_TABLE[offset:offset + (1 << k_skip)]))
    num, den = _lag_numden(sg, zg)
    return _lag_w(num, _batch_inv(den))


def _interpolate_at_z_on_lambda_g(values, k_skip: int, zg):
    """Σ_i L_i^Λ(z)·values[i] as a native ghash scalar."""
    w = _lagrange_weights(k_skip, zg, 1 << k_skip)
    return fnp.sum(w * ghash.to_ghash(fnp.asarray(values)))


def _interpolate_at_z_on_lambda(values, k_skip: int, zg) -> np.ndarray:
    """`_interpolate_at_z_on_lambda_g` serialized as uint64 lanes.

    values: uint64 [2^k_skip, 2]; zg: ghash scalar; returns uint64 [2]."""
    return ghash.from_ghash_host(
        _interpolate_at_z_on_lambda_g(values, k_skip, zg)
    )


def _interpolate_at_z_combined(values, k_skip: int, zg):
    """Evaluate the degree-<2·ell combined polynomial, zero on S, at `zg`.

    `values` are its ell evaluations on Λ. The omitted S evaluations are zero,
    so only the Λ Lagrange weights over the full S∪Λ domain contribute.
    """
    ell = 1 << k_skip
    nodes = ghash.to_ghash(fnp.asarray(_urm.PHI_8_TABLE[: 2 * ell]))
    selected = nodes[ell:]
    positions = fnp.arange(2 * ell)[None, :]
    selected_positions = fnp.arange(ell, 2 * ell)[:, None]
    same = positions == selected_positions
    num = _prod_axis1(
        fnp.where(
            same,
            _ONE_G,
            fnp.broadcast_to((zg + nodes)[None, :], (ell, 2 * ell)),
        )
    )
    den = _prod_axis1(
        fnp.where(same, _ONE_G, selected[:, None] + nodes[None, :])
    )
    weights = num * _batch_inv(den)
    return fnp.sum(weights * ghash.to_ghash(fnp.asarray(values)))


@frx.jit
def _fold_at_z(rows, w_g):
    """a_mlv[x_rest] = Σ_s witness[x_rest·ell + s]·L_s(z) (flock `fold_at_z_naive`),
    on device, reading the witness rows already resident from round1. rows: uint8
    [2^(m-k_skip), ell]; w_g: `binary_field_ghash [ell]` -> ghash [n_chunks].

    Select-and-XOR-reduce: the large `[n_chunks, ell]` intermediate is fused on the
    GPU instead of materialized in host numpy. The select drops to the lo/hi lanes —
    it zeroes a weight by integer-multiplying it against the 0/1 witness bit, which
    the field multiply can't express."""
    w = ghash.from_ghash(w_g)                                     # [ell, 2]
    masked = rows[:, :, None].astype(fnp.uint64) * w[None, :, :]  # 0 or w[s], uint64 [n,ell,2]
    return fnp.sum(ghash.to_ghash(masked), axis=1)                # ghash [n]
