# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Per-phase GPU prove timing with a hashes/second column, across hash circuits.

The e2e_*_ligerito_bench scripts each report one wall-clock number for one
circuit. That is not enough to steer throughput work: it cannot say which of
commit / zerocheck / lincheck / open owns the time, and it never converts to the
metric a throughput goal is stated in. This does both, for every Ligerito hash
circuit, off the goldens the byte gates already use. Pass `--throughput` for
the unsplit whole-prove number; omit it for the synchronized phase diagnostic.

It **reports hashes/second**: each circuit packs `n_sub * 2^(m - k_log)` hashes
into a proof, so cost per hash — the quantity a throughput target is about —
differs from cost per proof by a circuit-dependent constant.

It also refuses to run while another compute process holds the GPU, since a
neighbour saturating the SMs inflates a warm prove by ~28x on this box, and
when the selected ptxas predates clmad on a clmad-capable card, since the
shift/XOR fallback inflates the clmul-heavy phases ~15x and poisons the
per-wheel compile cache (`_ptxas.py`). Both are precondition checks, not
certificates: see the GPU-provenance section below for what they cannot see.

**One run of this is not a baseline.** The best-of-N below is within a single
process. Measured on an idle card, blake3 landed 13-19% apart *across*
processes at m <= 26, almost all of it inside `open`, which falls into distinct
clusters run to run while `zerocheck` reproduces to 2.5%. It is not thermal —
a back-to-back batch held 41-47C at pegged clocks with no throttle reason, and
wall time did not track temperature. So take several runs, report the spread,
and treat a single number at m <= 26 as having a wide error bar.

In diagnostic mode, **every phase is awaited before the next starts**, so the
split accounts for the whole prove and each phase is billed the work it actually
causes. Those barriers serialise work the `--throughput` path may overlap, so
the split is an upper bound rather than a headline throughput measurement.
`Sum` versus `wall` is printed as a self-check on the instrumentation: a gap
there means work escaped every phase.

Run (`CUDA_ROOT` must point at a **13.3** toolchain — `/usr/local/cuda` is not
necessarily one, and 12.9 silently selects the software GF(2^128) multiply,
worth ~5.5x on the whole prove; verify with `ptxas --version`. Do NOT reach for
`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` by default: at m32 it inflates the
prove ~16% and makes the phase-split mode OOM, and `PREALLOCATE=false` alone
is sufficient):
    export CUDA_ROOT=/usr/local/cuda
    export FRX_PLATFORMS=cuda,cpu FRX_ENABLE_X64=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    unset JAX_PLATFORMS JAX_ENABLE_X64
    export PATH="$CUDA_ROOT/bin:$PATH"
    PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \\
        python/flock_zorch/testing/prove_phase_bench.py [circuit ...] [options]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, NamedTuple

import frx

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
from zorch.grind import GRIND_WINDOW  # noqa: E402

from flock_zorch import lincheck, prover, zerocheck  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.r1cs_hashes import blake3_witness  # noqa: E402
from flock_zorch.testing._golden import unpack_bits  # noqa: E402
from flock_zorch.testing._ptxas import (  # noqa: E402
    clmad_ptxas_verdict,
    ptxas_version_text,
)
from flock_zorch.testing._util import await_all, best, best_of  # noqa: E402
from flock_zorch.types import ProveFastResult  # noqa: E402

PHASES = ("commit", "zerocheck", "lincheck", "open")


# The hash names the whole arm because the two shipped profiles pair their FS
# and Merkle choices: flock's is SHA-256 for both, the flock-challenge harness's
# is BLAKE3 for both. Mixed arms are expressible (`ProveProfile` takes the two
# separately, and one was measured while attributing the gap between these) but
# are not a configuration anything ships, so they are not offered here.
HASH_ARMS = {"sha256": "SHA256_PROFILE", "blake3": "BLAKE3_PROFILE"}


def _profile(args):
    """The arm this run measures.

    Changes WHICH bytes the proof is, not the protocol — both run the same
    reductions and the same open, so the two are directly comparable.

    It does NOT change what is timed. This harness measures witgen -> open and
    never serializes; the harness worker's window additionally includes
    `proof_io.bundle_bytes`. Picking `blake3` gets you the harness's arm, not
    its scope.
    """
    return getattr(prover, HASH_ARMS[args.hash])


class FoldGrindCost(NamedTuple):
    """What a config's fold PoW schedule costs, by two different measures.

    `expected_attempts` is what the difficulty implies — the protocol's own
    quantity. `windowed_hashes` is what zorch actually evaluates, which is the
    one that costs time, and the two differ by ~2x here. Reporting only the
    first invites the wrong conclusion about where the cost sits.
    """

    grinds: int
    expected_attempts: int
    windowed_hashes: int


def fold_grind_census(cfg, window: int = GRIND_WINDOW) -> FoldGrindCost:
    """Price the config's fold PoW schedule, mirroring
    `FlockChoreography.fold_grind_bits`: level `l`'s fold round `j` grinds
    `fold_grinding_bits[l] - j` bits, and only when that is > 0.

    Exists to price an asymmetry, not to describe the protocol. flock's
    `cuda-ghash/bench_ligerito` runs "grinding OFF" — it calls `grind_pow(0)`,
    the unconditional 0-bit query grind, and performs NO fold grinds. Our m32
    golden carries `grinding_bits [0]*6` (identical to theirs) but
    `fold_grinding_bits [19, 14, 11, 8, 6, 4]`, which is 21 real searches.
    Comparing the two provers' walls without pricing that compares different
    work.

    **The two totals differ, and the difference is the point.** `grind_search`
    tests a whole `GRIND_WINDOW`-wide counter batch per `while_loop` step, so a
    grind never costs less than one window however easy it is. At m32 that turns
    1.07M expected attempts into 2^21 = 2.10M hashes actually evaluated, and it
    moves where the work is: level 0 is 97% of the expected attempts but only
    53% of the windowed hashes, because 18 of the 21 sit at <= 16 bits and each
    still pays a full window — only the top three exceed it. A 0-bit grind is
    free (the transcripts special-case it to the canonical zero witness), which
    is why dropping only the fold schedule leaves the query grinds costing
    nothing.
    """
    ks = (cfg["initial_k"], *cfg["recursive_ks"])
    n = attempts = hashes = 0
    for level, k in enumerate(ks):
        for j in range(k):
            bits = cfg["fold_grinding_bits"][level] - j
            if bits > 0:
                n += 1
                attempts += 1 << bits
                hashes += max(window, 1 << bits)
    return FoldGrindCost(n, attempts, hashes)


def drop_fold_grinds(cfg) -> FoldGrindCost:
    """Zero `fold_grinding_bits` in `cfg`, returning what `fold_grind_census`
    reported before the edit.

    The query grinds are left alone: at 0 bits they still put a trivial nonce on
    the wire, which is exactly what flock's bench does, so zeroing only the fold
    schedule lands both provers on the same work. **The resulting proof is not
    gate-valid** — every challenge after a dropped grind moves — so this is for
    timing arms only.
    """
    before = fold_grind_census(cfg)
    cfg["fold_grinding_bits"] = [0] * len(cfg["fold_grinding_bits"])
    return before


def _phases(args):
    """Phase columns for this invocation: seed mode owns an extra leading
    `witgen` phase (seed->blocks->witness->stripe)."""
    return (("witgen",) + PHASES) if args.seed is not None else PHASES


# ---------------------------------------------------------------- GPU provenance
#
# What this records, and what it deliberately does not claim.
#
# It writes down what the card looked like, and refuses on one unambiguous
# fact: another compute process is on it. It does **not** certify that a
# measurement is trustworthy, and no output here should be read as doing so.
# It cannot see a host-side stall (which inflates every phase at once — the
# signature is all four moving together), non-compute graphics load, a
# neighbour that starts and exits between two samples, or the clock and
# thermal state.
#
# It also cannot see the largest source of error. On this box the same binary
# measured 13-19% apart across processes on an idle card at m <= 26, almost
# entirely inside `open` — coarser than most regressions worth benchmarking.
# A free card is a precondition for measuring, not evidence that a number is
# good; that comes from repeating the run and reporting the spread.


def _visible_gpu() -> str | None:
    """Physical GPU selected by a single numeric CUDA_VISIBLE_DEVICES entry."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return visible if visible.isdigit() else None


def _smi(query: str, gpu: str | None = None) -> str:
    cmd = ["nvidia-smi"]
    if gpu is not None:
        cmd += ["-i", gpu]
    cmd += [f"--query-{query}", "--format=csv,noheader,nounits"]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=15, check=True
    ).stdout.strip()


def gpu_provenance() -> tuple[str, int]:
    """`(card state for the record, count of other compute processes)`.

    The count is `-1` when the probe fails — no nvidia-smi, a CPU run, output
    drift — which never blocks a measurement.

    `memory.used` and `utilization.gpu` are card-wide and include this process
    (nvidia-smi attributes neither per process, and importing frx has already
    taken a context by the time anything here runs), so they are recorded and
    never compared against a threshold. Only the compute-app list names *other*
    processes, so it is the one thing worth acting on.
    """
    gpu = _visible_gpu()
    try:
        # One row per GPU. Aggregate rather than taking row 0: frx's device 0
        # need not be nvidia-smi's, and watching the wrong card silently is
        # worse than being occasionally too conservative.
        rows = [
            r.split(",")
            for r in _smi(
                "gpu=memory.used,memory.total,utilization.gpu", gpu
            ).splitlines()
            if r.strip()
        ]
        used = sum(int(r[0]) for r in rows)
        total = sum(int(r[1]) for r in rows)
        util = max(int(r[2]) for r in rows)
    except Exception as e:
        return f"state unknown ({type(e).__name__})", -1

    try:
        own = os.getpid()
        others = [
            p
            for p in (
                int(ln.split(",")[0])
                for ln in _smi("compute-apps=pid,used_memory", gpu).splitlines()
                if ln.strip()
            )
            if p != own
        ]
    except Exception:
        return f"{used}/{total} MiB, util {util}% (compute-app list unavailable)", -1

    who = f", {len(others)} other compute proc" if others else ", no other compute proc"
    return f"{used}/{total} MiB, util {util}%{who}", len(others)


# ------------------------------------------------------------------- circuits


def _csc(g):
    meta = g["meta"]
    return lincheck.CscCircuit(
        g["a0_rows"], g["b0_rows"], 1 << meta["k_log"], const_pin=meta["const_pin"]
    )


def _keccak3_circuit(_g):
    from flock_zorch.r1cs_hashes.keccak3_lincheck import Keccak3LincheckCircuit

    return Keccak3LincheckCircuit()


# name -> lincheck circuit builder. Everything else — golden filename, FS domain,
# dump example, loader, unpacker — follows from the name by the repo's own
# naming, so there is one place per circuit to get wrong.
CIRCUITS: dict[str, Callable] = {
    "blake3": _csc,
    "sha2": _csc,
    "keccak3": _keccak3_circuit,
}


@dataclass(frozen=True)
class Circuit:
    name: str

    @property
    def golden(self) -> str:
        return f"{self.name}_ligerito_golden.bin"

    @property
    def domain(self) -> bytes:
        return f"flock-{self.name}-lig-v0".encode()

    @property
    def n_sub(self) -> int:
        """Hashes packed into one 2^k_log block."""
        if self.name == "keccak3":
            from flock_zorch.r1cs_hashes.keccak3_lincheck import N_SUB

            return N_SUB  # 3 independent Keccak-f[1600] permutations per block
        return 1

    def build(self, g):
        return CIRCUITS[self.name](g)

    @property
    def _oracle(self):
        return importlib.import_module(
            f"flock_zorch.testing.{self.name}_ligerito_oracle_test"
        )

    def ingest(self, golden: str | None):
        """Load a golden through the gate's own loader, so the bench and the byte
        gate can never disagree about the wire. A missing file is reported by
        `_golden.open_golden`."""
        return self._oracle.load(golden or self.golden)

    def hashes_per_proof(self, meta) -> int:
        """Capacity, not occupancy: one proof commits 2^m bits laid out as
        2^(m - k_log) blocks of n_sub hashes each. A golden dumped with fewer
        hashes than that still pays the full proof cost, so throughput derived
        from a partly-filled golden understates the circuit (the keccak3 m=22
        golden holds 49 of 96 slots)."""
        return self.n_sub << (meta["m"] - meta["k_log"])


# -------------------------------------------------------------------- timing


def make_prove(circ: Circuit, g, unpacked: bool, seed: int | None = None, profile=None):
    """Returns a `prove(times) -> result` running one full prove.

    With `times`, every phase is awaited and recorded into it. There is exactly
    one definition of the sequence, and every statement lives inside a phase —
    inter-phase glue billed to nobody would make the split silently under-count
    the prove it claims to decompose.
    """
    profile = prover.SHA256_PROFILE if profile is None else profile
    meta, cfg = g["meta"], g["cfg"]
    m, k_log, k_skip = meta["m"], meta["k_log"], meta["k_skip"]
    ir = k_log - k_skip
    stmt, zlc = g["stmt"], g["zlc"]
    circuit = circ.build(g)

    seed_dev = None
    if seed is not None:
        # The witness comes from the device chain inside prove(); the golden
        # supplies only the circuit constants (cfg, statement digest, CSC
        # rows — all witness-independent). Only the 8-byte seed is uploaded.
        a_bits = b_bits = c_bits = z = zlc = None
        seed_dev = frx.device_put(fnp.uint64(seed))
    elif unpacked:
        witness = (
            unpack_bits(g["a"], m),
            unpack_bits(g["b"], m),
            unpack_bits(g["z"], m),
        )
        a_bits, b_bits, c_bits = (frx.device_put(x) for x in witness)
        z = frx.device_put(g["z"])
        zlc = lincheck.stripe_to_device(zlc, m, k_log)
    else:
        # Packed F128 — witness_to_rows unpacks on device (8x less host
        # transfer). Upload once — the lincheck stripe included. Left as host
        # numpy/bytes these re-cross PCIe every iteration, and the cost lands
        # on whichever phase touches them first — skewing the very split this
        # harness exists to report.
        witness = (g["a"], g["b"], g["z"])
        a_bits, b_bits, c_bits = (frx.device_put(x) for x in witness)
        z = frx.device_put(g["z"])
        zlc = lincheck.stripe_to_device(zlc, m, k_log)

    def prove(times=None):
        def phase(name, fn):
            if times is None:
                return fn()
            t0 = time.perf_counter()
            r = await_all(fn())
            times[name] = (time.perf_counter() - t0) * 1e3
            return r

        if seed_dev is None:
            wit_a, wit_b, wit_c, wit_z, wit_zlc = a_bits, b_bits, c_bits, z, zlc
        else:

            def _witgen():
                blocks = blake3_witness.blocks_from_seed(seed_dev, m - k_log)
                z3, a3, b3 = blake3_witness.witness_blake3(*blocks)
                zlc3 = blake3_witness.lincheck_stripe(z3)
                return a3.reshape(-1, 2), b3.reshape(-1, 2), z3.reshape(-1, 2), zlc3

            wit_a, wit_b, wit_c, wit_zlc = phase("witgen", _witgen)
            wit_z = wit_c

        def _commit():
            root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, wit_z, profile.tree)
            ch = profile.challenger_cls(circ.domain)
            prover.bind_statement(ch, stmt, root)
            return pdata, ch

        def _lincheck(zc):
            x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
            lc = lincheck.prove(
                wit_zlc, None, None, x_ab, m, k_log, k_skip, ch=ch, circuit=circuit
            )
            return x_ab, lc

        def _open(zc, x_ab, lc):
            ab = fnp.concatenate([lc.claim.r_inner_rest, x_ab.x_outer], axis=0)
            cc = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
            return prover.open_batch_ligerito(
                cfg,
                wit_z,
                pdata,
                [ab, cc],
                ch,
                profile.tree,
                precomputed_s_hat_vs=prover.ab_precomputed_s_hat_vs(
                    lc.z_vec, lc.claim.r_inner_rest
                ),
            )

        pdata, ch = phase("commit", _commit)
        # The claim, not the wire proof: `ZerocheckProof` holds wire fields
        # only, and the point the lincheck and open reduce (`z`,
        # `mlv_challenges`, `r_rest`) lives on `ZerocheckClaim`.
        zc_proof, zc = phase(
            "zerocheck",
            lambda: zerocheck.prove_packed(wit_a, wit_b, wit_c, m, ch=ch),
        )
        x_ab, lc = phase("lincheck", lambda: _lincheck(zc))
        opening = phase("open", lambda: _open(zc, x_ab, lc))
        return ProveFastResult(
            zerocheck=zc_proof,
            lincheck=(lc.rounds, lc.z_partial),
            pcs_open=opening,
            claim_ab_value=lc.claim.w,
            claim_c_value=zc.c_eval,
        )

    return prove


# ---------------------------------------------------------------------- main

# Prefix marking the machine-readable line under `--json`. The banner, the
# toolchain warnings and the fold-PoW notice all reach stdout too, so a consumer
# that assumed "the output is JSON" would have to parse around them; a marker
# keeps the human output intact and the machine output unambiguous.
JSON_MARK = "##bench-json## "


def _emit_json(circ, args, g, n_hash, wall, parts, dropped) -> None:
    """One `JSON_MARK`-prefixed line describing this measurement.

    Carries the config identity (`m`, `log_n`, the query ladder) alongside the
    timings because a wall compared against another prover's is only meaningful
    once both are shown to be the same instance, and a consumer that has to
    infer that from a filename will eventually infer it wrong.
    """
    meta, cfg = g["meta"], g["cfg"]
    payload = {
        "circuit": circ.name,
        "m": meta["m"],
        "log_n": meta["log_n"],
        "initial_k": cfg["initial_k"],
        "recursive_ks": list(cfg["recursive_ks"]),
        "queries": list(cfg["queries"]),
        "hashes": n_hash,
        "hash_arm": args.hash,
        "wall_ms": wall,
        "mode": "throughput" if args.throughput else "barriered",
        "phases": dict(parts) if parts else None,
        "fold_pow_dropped": (None if dropped is None else dropped._asdict()),
    }
    print(JSON_MARK + json.dumps(payload))


def bench(circ: Circuit, args) -> None:
    """Measure one circuit and print its row. Scoped to a function so the golden
    (~90 MB) and the circuit's device buffers are released before the next one."""
    g = circ.ingest(args.golden)
    meta = g["meta"]
    n_hash = circ.hashes_per_proof(meta)
    dropped = drop_fold_grinds(g["cfg"]) if args.no_fold_grind else None
    prove = make_prove(circ, g, args.unpacked, seed=args.seed, profile=_profile(args))
    phases = _phases(args)

    if dropped is not None:
        print(
            f"  fold PoW DROPPED: {dropped.grinds} grinds, "
            f"{dropped.expected_attempts:,} expected attempts = "
            f"{dropped.windowed_hashes:,} hashes actually evaluated "
            f"(GRIND_WINDOW={GRIND_WINDOW:,} floors every search). Matches "
            "flock's bench (query grinds at 0 bits remain, and cost nothing). "
            "This proof is NOT gate-valid — timing only."
        )

    if args.throughput:
        wall = best(lambda: prove(), args.runs)
        if args.json:
            _emit_json(circ, args, g, n_hash, wall, None, dropped)
            return
        print(
            f"{circ.name:>8} {meta['m']:>3} {n_hash:>8} "
            f"{wall:>9.2f}ms {n_hash * 1e3 / wall:>10.0f}"
        )
        if args.cpu_ms:
            print(
                f"  {args.cpu_ms / wall:.2f}x vs same-instance flock CPU "
                f"{args.cpu_ms:.0f}ms"
            )
        return

    def timed_prove():
        times = {}
        return prove(times), times

    wall, parts = best_of(timed_prove, args.runs)
    total = sum(parts.values())

    if args.json:
        _emit_json(circ, args, g, n_hash, wall, parts, dropped)
        return

    print(
        f"{circ.name:>8} {meta['m']:>3} {n_hash:>8} "
        + " ".join(f"{parts[p]:>9.2f}ms" for p in phases)
        + f" {total:>7.1f}ms {wall:>7.1f}ms {n_hash * 1e3 / wall:>10.0f}"
    )
    print("  " + "  ".join(f"{p} {100 * parts[p] / total:.0f}%" for p in phases))
    if args.cpu_ms:
        print(
            f"  {args.cpu_ms / wall:.2f}x vs same-instance flock CPU "
            f"{args.cpu_ms:.0f}ms"
        )
    if abs(total - wall) / wall > 0.10:
        print(
            f"  NOTE {wall - total:+.1f}ms "
            f"({100 * (wall - total) / wall:+.0f}%) of the prove is outside every "
            "phase — the split under-counts; instrumentation bug."
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("circuits", nargs="*", default=["blake3"], choices=list(CIRCUITS))
    ap.add_argument(
        "--golden",
        help="golden filename under artifacts/, for m-variant "
        "dumps (single circuit only)",
    )
    ap.add_argument("--runs", type=int, default=3, help="timed iterations, best-of")
    ap.add_argument(
        "--throughput",
        action="store_true",
        help="time the whole prove with one final synchronization; omit for the "
        "synchronized phase breakdown",
    )
    ap.add_argument(
        "--cpu-ms",
        type=float,
        help="flock CPU ms for the same instance "
        "(from bench_<circuit>_ligerito_cpu), to print "
        "a ratio; single circuit only",
    )
    ap.add_argument(
        "--unpacked",
        action="store_true",
        help="send witness as uint8 bits (8x host transfer) not packed F128",
    )
    ap.add_argument(
        "--seed",
        type=int,
        help="blake3 only: generate the witness on device from this challenge "
        "seed (seed->blocks->witness->stripe inside the timed window) instead "
        "of ingesting the golden's; the golden still supplies the circuit "
        "constants. The window is then snark.fast-comparable minus proof "
        "serialization.",
    )
    ap.add_argument(
        "--hash",
        choices=sorted(HASH_ARMS),
        default="sha256",
        help="Fiat-Shamir + Merkle hash: 'sha256' is flock's arm, the one every "
        "golden byte-gates; 'blake3' is the flock-challenge harness's. Swaps "
        "BOTH arms, unlike nsys_capture's --tree, which swaps the Merkle arm "
        "alone. Does not change what is timed — serialization is outside this "
        "window either way.",
    )
    ap.add_argument(
        "--no-fold-grind",
        action="store_true",
        help="zero the config's fold PoW schedule, leaving the 0-bit query "
        "grinds. This is what flock's cuda-ghash bench runs ('grinding OFF'), "
        "so it is the arm to use when comparing against it; at m32 it drops 21 "
        "searches worth ~1.07M expected hash attempts. The proof is NOT "
        "gate-valid under this flag — every challenge after a dropped grind "
        "moves. Timing arms only.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help=f"emit one machine-readable '{JSON_MARK.strip()}'-prefixed line "
        "per circuit instead of the table, for `rival_compare.py`",
    )
    ap.add_argument(
        "--allow-contended",
        action="store_true",
        help="measure even with another compute process on the card",
    )
    ap.add_argument(
        "--allow-clmadless-ptxas",
        action="store_true",
        help="measure even though the selected ptxas predates clmad "
        "(PTX ISA 9.3); the GF(2^128) multiply then runs as shift/XOR",
    )
    args = ap.parse_args()

    if args.seed is not None:
        if args.circuits != ["blake3"]:
            ap.error("--seed drives the device blake3 witgen; blake3 only")
        if args.unpacked:
            ap.error(
                "--seed and --unpacked conflict: the witness never "
                "crosses the host in seed mode"
            )

    if len(args.circuits) > 1:
        for flag, val in (("--golden", args.golden), ("--cpu-ms", args.cpu_ms)):
            if val is not None:
                ap.error(
                    f"{flag} describes one instance; pass a single circuit with it"
                )

    card, others = gpu_provenance()
    if others > 0 and not args.allow_contended:
        print(
            f"REFUSING to measure: {others} other compute process(es) on the card "
            f"({card}).\nA neighbour saturating the SMs inflates a warm prove by "
            "~28x on this box. Wait for the card, or pass --allow-contended for "
            "ratio-only work.",
            file=sys.stderr,
        )
        return 2

    device = frx.devices()[0]
    if device.platform == "gpu":
        reason = clmad_ptxas_verdict(
            ptxas_version_text(), getattr(device, "compute_capability", None)
        )
        if reason and not args.allow_clmadless_ptxas:
            print(
                f"REFUSING to measure: {reason} — zerocheck/open inflate ~15x "
                "and the wall reads as a regression, while the run poisons "
                "this wheel's XLA compile cache with the slow executables.\n"
                "Put a CUDA 13.3+ ptxas first on PATH (or set CUDA_DIR), wipe "
                "JAX_COMPILATION_CACHE_DIR if a mis-toolchained run populated "
                "it, or pass --allow-clmadless-ptxas for ratio-only work.",
                file=sys.stderr,
            )
            return 2

    print(f"device {device} | gpu: {card}")
    print(
        f"witness form: {'uint8 bits' if args.unpacked else 'packed F128'} "
        f"| best-of-{args.runs} within this process\n"
    )

    if not args.json:
        if args.throughput:
            hdr = f"{'circuit':>8} {'m':>3} {'hashes':>8} {'wall':>11} {'hash/s':>10}"
        else:
            hdr = (
                f"{'circuit':>8} {'m':>3} {'hashes':>8} "
                + " ".join(f"{p:>10}" for p in _phases(args))
                + f" {'sum':>9} {'wall':>9} {'hash/s':>10}"
            )
        print(hdr)
        print("-" * len(hdr))

    for name in args.circuits:
        bench(Circuit(name), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
