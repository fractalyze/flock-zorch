# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The snark.fast benchmark-profile prove body, shared between its two
consumers so they cannot drift: `bench_worker.py` runs it inside the harness
window, and `bench_ligerito_oracle_test.py` byte-gates its output against a
fork-verified bundle. One definition of the timed section — (circuit
constants, seed) → proof-file bytes."""

from __future__ import annotations

import frx
import frx.numpy as fnp

from flock_zorch import lincheck, proof_io, prover, zerocheck
from flock_zorch.pcs import ligerito as zorch_ligerito
from flock_zorch.r1cs_hashes import blake3_witness

# flock_benchmark_common::DOMAIN — the harness pins this, not flock's default.
BENCH_DOMAIN = b"flock-bench-v0"


def constants_golden(m: int) -> str:
    """The standard blake3 FLKBL golden carrying size-m circuit constants
    (cfg, statement digest, CSC rows — the witness-independent fields the
    benchmark-profile prove consumes)."""
    return (
        "blake3_ligerito_golden.bin" if m == 22 else f"blake3_ligerito_golden_m{m}.bin"
    )


class BenchProver:
    """The harness worker's per-process state: everything derivable from the
    circuit constants alone, built once so the timed call is only the
    seed-dependent chain."""

    def __init__(self, g: dict):
        meta = g["meta"]
        self._g = g
        self._m, self._k_log, self._k_skip = meta["m"], meta["k_log"], meta["k_skip"]
        self._csc = lincheck.CscCircuit(
            g["a0_rows"], g["b0_rows"], 1 << self._k_log, const_pin=meta["const_pin"]
        )
        self._params = proof_io.PcsParams(
            m=self._m,
            log_inv_rate=meta["lir"],
            log_batch_size=meta["lbs"],
            profile=proof_io.PROFILE_FAST,
            merkle_hash=proof_io.PARAMS_MERKLE_BLAKE3,
        )

    def prove_bundle(self, seed) -> bytes:
        """One benchmark-profile prove from the 8-byte seed: the device
        witness chain, the reductions under `prover.BENCHMARK_PROFILE` on the
        harness domain, the batched Ligerito open, and the wire serialization
        (which pulls every proof field to host). The seed is traced, so a
        warm-up call at the same log2 compiles every program the timed call
        runs."""
        g, m, k_log, k_skip = self._g, self._m, self._k_log, self._k_skip
        ir = k_log - k_skip
        profile = prover.BLAKE3_PROFILE

        seed_dev = frx.device_put(fnp.uint64(seed))
        z, a, b = blake3_witness.witness_blake3(
            *blake3_witness.blocks_from_seed(seed_dev, m - k_log)
        )
        zlc = blake3_witness.lincheck_stripe(z)
        z, a, b = (x.reshape(-1, 2) for x in (z, a, b))

        root, pdata = zorch_ligerito.commit_flock_ligerito(g["cfg"], z, profile.tree)
        ch = profile.challenger_cls(BENCH_DOMAIN)
        prover.bind_statement(ch, g["stmt"], root)
        zc_proof, zc = zerocheck.prove_packed(a, b, z, m, ch=ch)
        x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
        lc_rounds, lc_zp, lc_claim = lincheck.prove(
            zlc, None, None, x_ab, m, k_log, k_skip, ch=ch, circuit=self._csc
        )
        assert lc_claim is not None, "full lincheck prove always yields a claim"
        ab_full = fnp.concatenate([lc_claim.r_inner_rest, x_ab.x_outer], axis=0)
        c_full = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
        out = prover.open_batch_ligerito(
            g["cfg"], z, pdata, [ab_full, c_full], ch, profile.tree
        )

        return proof_io.bundle_bytes(
            root,
            self._params,
            zc_proof,
            lc_rounds,
            lc_zp,
            out.ring_switches,
            out.ligerito,
        )
