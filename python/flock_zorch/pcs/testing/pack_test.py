# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Pure-host gates for the witness packers, checked against the layouts their frx
consumers document — `prover._unpack_bits` (z_packed) and
`lincheck.partial_fold_packed_z` (the lincheck bytes). No golden, no field ops."""
from __future__ import annotations

import sys

import numpy as np

from flock_zorch.pcs.pack import (
    _unpack_flat,
    pack_witness,
    pack_z_lincheck_from_packed,
)


def _rand_bits(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)


def _report(name: str, ok: bool) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def _check_roundtrip(m: int) -> bool:
    w = _rand_bits(m, 1 << m)
    return _report(f"pack↔unpack roundtrip (m={m})",
                   np.array_equal(_unpack_flat(pack_witness(w, m)), w))


def _check_lsb_first() -> bool:
    # element 0 covers bits [0..128): bit i → 2^i in lo (i<64) / hi (i≥64).
    w = np.zeros(128, np.uint8)
    w[0] = w[3] = w[64] = 1
    lane = pack_witness(w, 7)
    return _report("LSB-first lo/hi split",
                   int(lane[0, 0]) == 0b1001 and int(lane[0, 1]) == 1)


def _check_wrong_length_raises() -> bool:
    try:
        pack_witness(np.zeros(10, np.uint8), 4)
    except ValueError:
        return _report("pack_witness rejects wrong length", True)
    return _report("pack_witness rejects wrong length", False)


def _check_lincheck_layout(m: int, k_log: int) -> bool:
    # Read the lincheck bytes exactly as partial_fold_packed_z ingests them:
    # reshape (n_bytes, k); bit r of byte[byte_idx, i_inner] == z[i_inner + (8·byte_idx + r)·k].
    w = _rand_bits(m * 100 + k_log, 1 << m)
    zlc = pack_z_lincheck_from_packed(pack_witness(w, m), m, k_log)
    k = 1 << k_log
    n_bytes = (1 << m) // k // 8
    stripes = np.frombuffer(zlc, np.uint8).reshape(n_bytes, k)
    ok = True
    for byte_idx in range(min(n_bytes, 4)):
        for i_inner in range(min(k, 6)):
            byte = int(stripes[byte_idx, i_inner])
            for r in range(8):
                want = int(w[i_inner + (8 * byte_idx + r) * k])
                ok = ok and ((byte >> r) & 1) == want
    return _report(f"lincheck byte layout (m={m}, k_log={k_log})", ok)


def _check_lincheck_guards() -> bool:
    zp = pack_witness(_rand_bits(1, 1 << 13), 13)
    ok = len(pack_z_lincheck_from_packed(zp, 13, 6)) == (1 << 13) // 8
    try:
        pack_z_lincheck_from_packed(zp, 13, 11)  # n_outer = 4, not byte-aligned
        ok = False
    except ValueError:
        pass
    return _report("lincheck length + byte-alignment guards", ok)


def main() -> int:
    ok = True
    for m in (7, 8, 13, 16):
        ok = _check_roundtrip(m) and ok
    ok = _check_lsb_first() and ok
    ok = _check_wrong_length_raises() and ok
    for m, k_log in ((13, 6), (16, 10), (16, 8)):
        ok = _check_lincheck_layout(m, k_log) and ok
    ok = _check_lincheck_guards() and ok
    print(f"witness packers: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
