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


def _interpolate_at_z_on_lambda(values, k_skip: int, zg) -> np.ndarray:
    """Σ_i L_i^Λ(z)·values[i] (flock `interpolate_at_z_on_lambda`).

    values: uint64 [2^k_skip, 2]; zg: ghash scalar; returns uint64 [2] (a proof field)."""
    w = _lagrange_weights(k_skip, zg, 1 << k_skip)
    prod = w * ghash.to_ghash(fnp.asarray(values))
    return ghash.from_ghash_host(fnp.sum(prod))  # XOR-sum inner product


@frx.jit
def _fold_unpacked_at_z(rows, w_g):
    """a_mlv[x_rest] = Σ_s witness[x_rest·ell + s]·L_s(z) (flock `fold_at_z_naive`),
    on device, reading the witness rows already resident from round1. rows: uint8
    [2^(m-k_skip), ell]; w_g: `binary_field_ghash [ell]` -> ghash [n_chunks].

    The zorch reduction streams fixed row blocks through Pallas, bounding the
    selector temporary independently of the witness size."""
    if frx.default_backend() == "cpu":
        w = ghash.from_ghash(w_g)
        masked = rows[:, :, None].astype(fnp.uint64) * w[None, :, :]
        return fnp.sum(ghash.to_ghash(masked), axis=1)
    # Keep Pallas out of CPU-only Bazel runfiles.  The GPU venv carries Mosaic;
    # importing it under the hermetic CPU wheel needlessly initializes that
    # optional backend before this fallback can run.
    from zorch.utils import binary_field as bf
    return bf.bit_select_xor_reduce(rows, w_g)


@frx.jit
def _fold_packed_at_z(packed, w_g):
    """Fold a packed F128 witness without materializing its uint8 bit rows.

    Each packed uint64 lane becomes eight selector bytes.  zorch builds one
    256-entry XOR table per byte position, then gathers eight entries per output
    row — flock-core's ``UniSkipFoldTable`` strategy on the GPU."""
    rows = frx.lax.bitcast_convert_type(packed, fnp.uint8).reshape(-1, 8)
    if frx.default_backend() == "cpu":
        bit = fnp.arange(8, dtype=fnp.uint8)
        selectors = ((rows[:, :, None] >> bit) & fnp.uint8(1)).reshape(-1, 64)
        return _fold_unpacked_at_z(selectors, w_g)
    from zorch.utils import binary_field as bf
    return bf.byte_select_xor_reduce(rows, w_g)


def _fold_at_z(rows, w_g):
    """Dispatch to the packed-byte fold when the original witness is available."""
    if (getattr(rows, "ndim", 0) == 2 and rows.shape[-1] == 2
            and np.dtype(rows.dtype) == np.uint64):
        return _fold_packed_at_z(rows, w_g)
    return _fold_unpacked_at_z(rows, w_g)
