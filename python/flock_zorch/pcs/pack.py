# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Host-side bit-witness packing — the frx port of flock-core `pcs::pack::pack_witness`
and `lincheck::pack_z_lincheck_from_packed`.

`prove_fast` reads the witness in two serializations: `z_packed` (F₂¹²⁸ lanes, for
the Ligerito commit + zerocheck) and the lincheck witness bytes (for
`lincheck.partial_fold_packed_z`). Both are cuts of ONE flat 2^m-bit witness at
logical index `i_inner + i_outer·k`, so they encode the same z — which is why the
byte packer derives from `z_packed` rather than re-reading the bits. The port ships
only the consumers (`prover._unpack_bits`, `partial_fold_packed_z`), so callers have
had to ingest both from a flock golden; these packers let an instance be built in
Python instead. Pure host NumPy: witness prep runs once, off device.
"""
from __future__ import annotations

import numpy as np

# F₂¹²⁸ holds 128 = 2^7 bits, so one element packs 2^LOG_PACKING witness coords.
LOG_PACKING = 7
PACKING_WIDTH = 1 << LOG_PACKING


def pack_witness(z_bits, m: int) -> np.ndarray:
    """Flat Boolean witness `z` [2^m] → F₂¹²⁸ lanes uint64 [2^(m-7), 2] (lo, hi):
    bit i of element e is z[e·128 + i], LSB-first (i<64 in lo, else hi). The host
    inverse of `prover._unpack_bits`; flock `pcs::pack::pack_witness`."""
    z = np.ascontiguousarray(z_bits, dtype=np.uint8).reshape(-1)
    if z.size != 1 << m:
        raise ValueError(f"z length must be 2^m = {1 << m}, got {z.size}")
    if m < LOG_PACKING:
        raise ValueError(f"witness too small to pack: m={m} < LOG_PACKING={LOG_PACKING}")
    # [e, half, r] = z[e·128 + half·64 + r]; bit r of each half weighted by 2^r.
    halves = z.reshape(1 << (m - LOG_PACKING), 2, 64).astype(np.uint64)
    weights = np.uint64(1) << np.arange(64, dtype=np.uint64)
    return (halves * weights).sum(axis=2).astype(np.uint64)


def pack_z_lincheck_from_packed(z_packed, m: int, k_log: int) -> bytes:
    """F₂¹²⁸-lane witness → lincheck witness bytes. `byte[byte_idx·k + i_inner]`
    carries at bit r the witness bit at logical index `i_inner + (8·byte_idx + r)·k`
    (k = 2^k_log) — flock `lincheck::pack_z_lincheck_from_packed`, the byte layout
    `lincheck.partial_fold_packed_z` reshapes to `(n_bytes, k)` and folds."""
    zp = np.ascontiguousarray(z_packed, dtype=np.uint64).reshape(-1, 2)
    n_total = 1 << m
    if zp.shape[0] != n_total // PACKING_WIDTH:
        raise ValueError(
            f"z_packed must be [2^(m-7), 2] = [{n_total // PACKING_WIDTH}, 2], got {zp.shape}"
        )
    k = 1 << k_log
    n_outer = n_total // k
    if n_outer % 8:
        raise ValueError(f"need n_outer ({n_outer}) divisible by 8 for byte stripes")
    # grid[i_outer, i_inner] = z[i_inner + i_outer·k]; each output byte packs 8
    # consecutive i_outer (one stripe) at a fixed i_inner, bit r = i_outer & 7.
    grid = _unpack_flat(zp).reshape(n_outer, k)
    stripes = grid.reshape(n_outer // 8, 8, k)  # [byte_idx, r, i_inner]
    weights = (np.uint8(1) << np.arange(8, dtype=np.uint8)).reshape(1, 8, 1)
    out = (stripes * weights).sum(axis=1).astype(np.uint8)  # [n_bytes, k]
    return out.reshape(-1).tobytes()


def _unpack_flat(z_packed) -> np.ndarray:
    """Host inverse of `pack_witness`: F₂¹²⁸ lanes [n, 2] → flat bits [n·128] uint8.
    The NumPy twin of `prover._unpack_bits` (which runs on device)."""
    zp = np.ascontiguousarray(z_packed, dtype=np.uint64).reshape(-1, 2)
    r = np.arange(64, dtype=np.uint64)
    lo = ((zp[:, 0:1] >> r) & 1).astype(np.uint8)
    hi = ((zp[:, 1:2] >> r) & 1).astype(np.uint8)
    return np.concatenate([lo, hi], axis=1).reshape(-1)
