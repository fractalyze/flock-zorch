# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The snark.fast benchmark-profile prove body, shared between its three
consumers so they cannot drift: `bench_worker.py` runs it inside the harness
window, `bench_ligerito_oracle_test.py` byte-gates its output against a
fork-verified bundle, and `cpu_phase_split.py` attributes its wall time. One
definition of the timed section — (circuit constants, seed) → proof-file
bytes."""

from __future__ import annotations

import time

import frx
import frx.numpy as fnp
from frx import profiler

from flock_zorch import lincheck, proof_io, prover, zerocheck
from flock_zorch.pcs import ligerito as zorch_ligerito
from flock_zorch.r1cs_hashes import blake3_witness
from flock_zorch.testing._util import await_all

# Phase columns of `prove_bundle`, in execution order. `serialize` is the
# harness's own tail — `bundle_bytes` pulls every proof field to host — and is
# inside the timed window, unlike `prove_phase_bench`'s witgen→open scope.
PHASES = ("witgen", "commit", "zerocheck", "lincheck", "open", "serialize")

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

    def prove_bundle(self, seed, times: dict | None = None) -> bytes:
        """One benchmark-profile prove from the 8-byte seed: the device
        witness chain, the reductions under `prover.BLAKE3_PROFILE` on the
        harness domain, the batched Ligerito open, and the wire serialization
        (which pulls every proof field to host). The seed is traced, so a
        warm-up call at the same log2 compiles every program the timed call
        runs.

        With `times`, each `PHASES` entry is awaited and its wall recorded
        there in ms. The barriers serialise work the untimed path may overlap,
        so a split is an upper bound on each phase and its sum exceeds the
        `times=None` wall — read it for attribution, never as the throughput
        number. Every statement lives inside a phase, so the split cannot
        silently under-count the prove it decomposes.
        """
        g, m, k_log, k_skip = self._g, self._m, self._k_log, self._k_skip
        ir = k_log - k_skip
        profile = prover.BLAKE3_PROFILE

        def phase(name, fn):
            if times is None:
                return fn()
            # The annotation is what lets a trace attribute an HLO op to a
            # phase: op events and annotation events share one clock, so
            # `cpu_phase_split --mode ops` buckets by timestamp rather than
            # guessing a module->phase map. It costs nothing untraced.
            t0 = time.perf_counter()
            with profiler.TraceAnnotation(name):
                r = await_all(fn())
            times[name] = (time.perf_counter() - t0) * 1e3
            return r

        def _witgen():
            seed_dev = frx.device_put(fnp.uint64(seed))
            z3, a3, b3 = blake3_witness.witness_blake3(
                *blake3_witness.blocks_from_seed(seed_dev, m - k_log)
            )
            zlc3 = blake3_witness.lincheck_stripe(z3)
            return (*(x.reshape(-1, 2) for x in (z3, a3, b3)), zlc3)

        def _commit():
            root, pdata = zorch_ligerito.commit_flock_ligerito(
                g["cfg"], z, profile.tree
            )
            ch = profile.challenger_cls(BENCH_DOMAIN)
            prover.bind_statement(ch, g["stmt"], root)
            return root, pdata, ch

        def _lincheck(zc):
            x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
            lc = lincheck.prove(
                zlc, None, None, x_ab, m, k_log, k_skip, ch=ch, circuit=self._csc
            )
            assert lc.claim is not None, "full lincheck prove always yields a claim"
            return x_ab, lc

        def _open(zc, x_ab, lc):
            ab_full = fnp.concatenate([lc.claim.r_inner_rest, x_ab.x_outer], axis=0)
            c_full = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
            return prover.open_batch_ligerito(
                g["cfg"],
                z,
                pdata,
                [ab_full, c_full],
                ch,
                profile.tree,
                precomputed_s_hat_vs=prover.ab_precomputed_s_hat_vs(
                    lc.z_vec, lc.claim.r_inner_rest
                ),
            )

        z, a, b, zlc = phase("witgen", _witgen)
        root, pdata, ch = phase("commit", _commit)
        zc_proof, zc = phase(
            "zerocheck", lambda: zerocheck.prove_packed(a, b, z, m, ch=ch)
        )
        x_ab, lc = phase("lincheck", lambda: _lincheck(zc))
        out = phase("open", lambda: _open(zc, x_ab, lc))

        return phase(
            "serialize",
            lambda: proof_io.bundle_bytes(
                root,
                self._params,
                zc_proof,
                lc.rounds,
                lc.z_partial,
                out.ring_switches,
                out.ligerito,
            ),
        )
