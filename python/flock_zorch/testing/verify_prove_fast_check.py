# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""GPU gate: `verifier.verify` accepts a `prove_fast` proof and rejects tampered ones.

Builds the identity R1CS in Python (packers, no golden), proves it, then runs the
full verify — bind → zerocheck → lincheck → ring-switch → Ligerito open — on a fresh
challenger. Not a bazel target (frx GPU plugin + the Ligerito recursion). Run:
    cd .../flock-zorch.worktrees/agent0
    CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" .venv/bin/python \
        python/flock_zorch/testing/verify_prove_fast_check.py
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch import ghash, prover, verifier  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.pcs import ligerito as zlig  # noqa: E402
from flock_zorch.pcs.pack import pack_witness, pack_z_lincheck_from_packed  # noqa: E402

_M, _K_LOG, _K_SKIP, _DOMAIN = 13, 8, 6, b"flock-verify-check"
_CFG = dict(initial_k=2, recursive_ks=[2], log_inv_rates=[1, 2], queries=[4, 3],
            grinding_bits=[1, 0], fold_grinding_bits=[1, 0], ood_samples=[0, 1],
            recursive_steps=1)


def _flip(x):
    lanes = np.asarray(ghash.to_lanes(x)).copy().reshape(2)
    lanes[0] ^= np.uint64(1)
    return lanes


def _verify(res, root, eye, stmt):
    return bool(verifier.verify(_CFG, root, stmt, res, eye, eye, _M, _K_LOG, _K_SKIP,
                                Challenger(_DOMAIN)))


def main() -> int:
    z_bits = np.random.default_rng(0).integers(0, 2, 1 << _M).astype(np.uint8)
    z_packed = pack_witness(z_bits, _M)
    z_lincheck = pack_z_lincheck_from_packed(z_packed, _M, _K_LOG)
    eye = np.eye(1 << _K_LOG, dtype=np.uint64)
    stmt = bytes(range(32))

    res = prover.prove_fast(z_packed, _M, _K_LOG, _K_SKIP, eye, eye, z_lincheck, stmt,
                            _CFG, domain=_DOMAIN)
    root = zlig.commit_flock_ligerito(_CFG, z_packed)[0]

    ok = True
    accept = _verify(res, root, eye, stmt)
    ok = ok and accept
    print(f"  {'PASS' if accept else 'FAIL'}  accept honest proof")

    # Tamper the zerocheck claim → the sumcheck chain rejects.
    zc_bad = dataclasses.replace(res.zerocheck, final_a_eval=_flip(res.zerocheck.final_a_eval))
    t1 = not _verify(dataclasses.replace(res, zerocheck=zc_bad), root, eye, stmt)
    ok = ok and t1
    print(f"  {'PASS' if t1 else 'FAIL'}  reject tampered zerocheck eval")

    # Tamper a ring-switch message → the opening check rejects.
    rs = list(res.pcs_open.ring_switches)
    rs[0] = ghash.to_ghash(frx.numpy.asarray(_tamper_slice(rs[0])))
    open_bad = dataclasses.replace(res.pcs_open, ring_switches=rs)
    t2 = not _verify(dataclasses.replace(res, pcs_open=open_bad), root, eye, stmt)
    ok = ok and t2
    print(f"  {'PASS' if t2 else 'FAIL'}  reject tampered PCS opening")

    print(f"prove_fast → verify: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _tamper_slice(s_hat_v):
    lanes = np.asarray(ghash.to_lanes(s_hat_v)).copy()
    lanes[0, 0] ^= np.uint64(1)
    return lanes


if __name__ == "__main__":
    sys.exit(main())
