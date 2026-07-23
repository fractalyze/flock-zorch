"""Device precompute for zerocheck's round-1 c-claim and round-2 binding — the
Lagrange-weight / fold-at-z cluster, kept out of zerocheck.py so prove_packed
reads as the PIOP. Byte-identical to flock's zerocheck/{univariate_skip,
multilinear}: the basis machinery is zorch's `compute_lagrange_basis` (Lagrange
values are unique field elements, so the wire cannot move); only the φ₈ node
selection is flock's.

Requires jax_enable_x64.
"""
from __future__ import annotations

import numpy as np
import frx
import frx.numpy as fnp

from flock_zorch import ghash
from flock_zorch.zerocheck import _urm
from zorch.poly.univariate import compute_lagrange_basis


def _lagrange_weights(k_skip: int, zg, offset: int):
    """L_i(z) over the φ₈-embedded nodes PHI_8_TABLE[offset+i], i∈[0, 2^k_skip).
    offset=0 → the S domain; offset=2^k_skip → the Λ domain.

    zorch's `compute_lagrange_basis` (one jitted kernel: masked num/den products
    + the dtype's native divide); flock only picks the φ₈ nodes.

    zg: `binary_field_ghash` scalar (z is a value in the L_i formula, not an index).
    Returns `binary_field_ghash [2^k_skip]` — never leaves the dtype."""
    sg = ghash.to_ghash(fnp.asarray(_urm.PHI_8_TABLE[offset:offset + (1 << k_skip)]))
    return compute_lagrange_basis(zg, sg)


def _interpolate_at_z_on_lambda(values, k_skip: int, zg):
    """Σ_i L_i^Λ(z)·values[i] (flock `interpolate_at_z_on_lambda`).

    values: `binary_field_ghash [2^k_skip]`; zg: ghash scalar; returns a ghash
    scalar (stays device-resident — byte-gate readers lift via `ghash.to_lanes`)."""
    w = _lagrange_weights(k_skip, zg, 1 << k_skip)
    return fnp.sum(w * values)  # XOR-sum inner product


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
    return bf.bit_select_xor_reduce(rows, w_g, reduce="bits")


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
            and rows.dtype == np.uint64):
        return _fold_packed_at_z(rows, w_g)
    return _fold_unpacked_at_z(rows, w_g)
