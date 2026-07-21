# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Round-trip gate for the zerocheck verifier (no golden): `prove_packed` then
`verifier.verify` accepts and its reconstructed claim matches the prover's, and a
tampered proof is rejected with `ok=False`. Software-mul path, so it runs on CPU
under bazel alongside the oracle gates."""
from __future__ import annotations

import dataclasses
import sys

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch import ghash, zerocheck  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.zerocheck import verifier  # noqa: E402

DOMAIN = b"flock-zc-verify-test"


def _lanes(x) -> np.ndarray:
    return np.asarray(ghash.to_lanes(x))


def _eq(a, b) -> bool:
    return bool(np.array_equal(_lanes(a), _lanes(b)))


def _report(name: str, ok: bool) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def _witness(m: int, seed: int):
    rng = np.random.default_rng(seed)
    nbytes = (1 << m) // 8

    def bits():
        return np.unpackbits(rng.integers(0, 256, nbytes, dtype=np.uint8), bitorder="little")

    a, b = bits(), bits()
    return a, b, a & b  # honest witness: a·b = c


def _check(m: int) -> bool:
    a, b, c = _witness(m, m)
    proof = zerocheck.prove_packed(a, b, c, m, DOMAIN)
    claim, _, ok = verifier.verify(m, proof, Challenger(DOMAIN))
    good = (
        bool(ok)
        and _eq(claim.z, proof.z)
        and _eq(claim.mlv_challenges, proof.mlv_challenges)
        and _eq(claim.r_rest, proof.r_rest)
        and _eq(claim.a_eval, proof.final_a_eval)
        and _eq(claim.c_eval, proof.final_c_eval)
    )
    accept = _report(f"accept + claim reconstruction (m={m})", good)

    # Corrupt the final â eval — the sumcheck identity must now fail.
    bad_a = _lanes(proof.final_a_eval).copy().reshape(2)
    bad_a[0] ^= np.uint64(1)
    _, _, ok_bad = verifier.verify(m, dataclasses.replace(proof, final_a_eval=bad_a),
                                   Challenger(DOMAIN))
    reject = _report(f"tamper rejected (m={m})", not bool(ok_bad))
    return accept and reject


def main() -> int:
    ok = True
    for m in (13, 14, 16):
        ok = _check(m) and ok
    print(f"zerocheck verifier: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
