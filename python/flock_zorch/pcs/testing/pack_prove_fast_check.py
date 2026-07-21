# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""GPU integration gate: the witness packers actually drive `prover.prove_fast`.

Builds an identity R1CS (A₀=B₀=C₀=I, so a=b=c=z) entirely in Python — the witness
packed by `pcs.pack`, no flock golden — and runs the fused four-stage prover
(Ligerito commit → zerocheck → lincheck → batched Ligerito open). Asserts the
proof's stage shapes, which pin the round counts to (m, k_log, k_skip). This is
the end-to-end complement to the pure-host `pack_test`: `pack_test` proves the
byte layouts, this proves they are what `prove_fast` consumes.

Not a bazel target (needs the frx GPU plugin + a ~2 GB dense identity at m=22).
Run on the venv:
    export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
    PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" .venv/bin/python \
        python/flock_zorch/pcs/testing/pack_prove_fast_check.py
"""
from __future__ import annotations

import sys

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch import prover  # noqa: E402
from flock_zorch.pcs.pack import pack_witness, pack_z_lincheck_from_packed  # noqa: E402

# The m22_fast / log_n=15 flock Ligerito profile (fold_ks=(6,4,3), residual 2).
_M, _K_LOG, _K_SKIP = 22, 14, 6
_CFG = dict(initial_k=6, recursive_ks=[4, 3], log_inv_rates=[1, 2, 4],
            queries=[148, 100, 60], grinding_bits=[2, 1, 0],
            fold_grinding_bits=[3, 2, 0], ood_samples=[0, 1, 1], recursive_steps=2)


def main() -> int:
    # Any bit witness is a satisfying identity instance: the zerocheck gate is
    # z∘z=z, and b² = b for every bit.
    w = np.random.default_rng(0).integers(0, 2, 1 << _M).astype(np.uint8)
    z_packed = pack_witness(w, _M)
    z_lincheck = pack_z_lincheck_from_packed(z_packed, _M, _K_LOG)
    k = 1 << _K_LOG
    a0 = np.eye(k, dtype=np.uint64)
    b0 = np.eye(k, dtype=np.uint64)
    stmt = bytes(range(32))  # opaque instance digest; the prover only observes it

    res = prover.prove_fast(z_packed, _M, _K_LOG, _K_SKIP, a0, b0, z_lincheck, stmt, _CFG)

    checks = {
        "zerocheck multilinear rounds == m - k_skip":
            len(res.zerocheck.multilinear_rounds) == _M - _K_SKIP,
        "lincheck rounds == k_log - k_skip":
            len(res.lincheck[0]) == _K_LOG - _K_SKIP,
        "pcs_open ring_switches == 2 (ab + c)":
            len(res.pcs_open.ring_switches) == 2,
        "ab/c claims present":
            res.claim_ab_value is not None and res.claim_c_value is not None,
    }
    ok = True
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    print(f"packers → prove_fast (identity m={_M}): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
