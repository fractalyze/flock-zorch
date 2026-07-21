# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Round-trip gate for the lincheck verifier (no golden): prove zerocheck then
lincheck on one challenger, verify both on a fresh one, and check the lincheck
claim matches and a tampered z_partial is rejected. Software-mul → runs on CPU."""
from __future__ import annotations

import sys

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch import ghash, lincheck, zerocheck  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.lincheck import verifier as lcv  # noqa: E402
from flock_zorch.lincheck.prover import AbClaimPoint  # noqa: E402
from flock_zorch.pcs.pack import pack_witness, pack_z_lincheck_from_packed  # noqa: E402
from flock_zorch.zerocheck import verifier as zcv  # noqa: E402

DOMAIN = b"flock-lc-verify-test"


def _eq(a, b) -> bool:
    return bool(np.array_equal(np.asarray(ghash.to_lanes(a)), np.asarray(ghash.to_lanes(b))))


def _report(name: str, ok: bool) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def _check(m: int, k_log: int, k_skip: int) -> bool:
    z_bits = np.random.default_rng(m).integers(0, 2, 1 << m).astype(np.uint8)
    z_packed = pack_witness(z_bits, m)
    z_lincheck = pack_z_lincheck_from_packed(z_packed, m, k_log)
    eye = np.eye(1 << k_log, dtype=np.uint64)

    chp = Challenger(DOMAIN)
    zc = zerocheck.prove_packed(z_bits, z_bits, z_bits, m, ch=chp)
    x_ab = AbClaimPoint.from_zerocheck(zc, k_log - k_skip)
    lp = lincheck.prove(z_lincheck, eye, eye, x_ab, m, k_log, k_skip, ch=chp, capture=True)

    chv = Challenger(DOMAIN)
    zcv.verify(m, zc, chv)
    claim, _, ok = lcv.verify(m, k_log, k_skip, eye, eye, x_ab,
                              zc.final_a_eval, zc.final_b_eval, lp, chv)
    good = (
        bool(ok)
        and _eq(claim.w, lp.claim.w)
        and _eq(claim.r_inner_skip, lp.claim.r_inner_skip)
        and _eq(claim.r_inner_rest, lp.claim.r_inner_rest)
    )
    accept = _report(f"accept + claim reconstruction (m={m}, k_log={k_log})", good)

    zp = np.asarray(ghash.to_lanes(lp.z_partial)).copy()
    zp[0, 0] ^= np.uint64(1)
    bad = lp._replace(z_partial=ghash.to_ghash(frx.numpy.asarray(zp)))
    chv2 = Challenger(DOMAIN)
    zcv.verify(m, zc, chv2)
    _, _, ok_bad = lcv.verify(m, k_log, k_skip, eye, eye, x_ab,
                              zc.final_a_eval, zc.final_b_eval, bad, chv2)
    reject = _report(f"tamper rejected (m={m}, k_log={k_log})", not bool(ok_bad))
    return accept and reject


def main() -> int:
    ok = True
    for m, k_log, k_skip in ((13, 8, 6), (14, 9, 6)):
        ok = _check(m, k_log, k_skip) and ok
    print(f"lincheck verifier: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
