# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Where the FRX CPU tier's wall time goes, inside the harness's own window.

`prove_phase_bench` answers the same question for the GPU tier, and cannot be
pointed at this one: its window is witgen→open on the sha256 arm, while the
snark.fast harness times `BenchProver.prove_bundle` — the blake3 arm, plus the
`serialize` tail that pulls every proof field to host. This drives that exact
body (`_bench_profile.PHASES`), so a share here is a share of the number the
leaderboard scores.

Three modes, because the CPU tier's cost splits along two axes that a single
table conflates:

- `wall`   — the unbarriered prove, min-of-N with spread. The headline.
- `split`  — the same prove with every phase awaited, min-of-N with spread.
             Barriers serialise work the `wall` path overlaps, so the sum
             exceeds the `wall` number; this ranks phases, it does not price
             the prove.
- `ops`    — one traced prove, per-HLO-op time rolled up into kernel classes
             (`KERNEL_CLASSES`), bucketed by phase from the trace's own
             annotations. Answers which *codegen* owns a phase, which is the
             question a phase name cannot: `zerocheck` is a binary-field
             select + XOR-reduce (prime-ir's lowering), a GHASH multiply, and
             an NTT transform (XLA:CPU's parallelism) in one, and the three
             move under different repositories.

**Startup is not part of the prove and is reported apart from it.** The harness
spawns a fresh worker per trial, so each trial repays imports, golden load and
XLA deserialization against the 300 s readiness budget; folding that into a
per-hash cost would attribute compiler-cache work to the prover. `wall` prints
both, separately.

Tracing inflates: read shares from `ops` and absolute times from `wall`, never
the reverse. The `ops` header prints its own traced wall next to the clean one
so the inflation is visible rather than assumed.

Run (CPU tier, physical cores only — SMT costs a few percent on this box, and
sibling pairing is read from topology, not assumed):

    export FRX_PLATFORMS=cpu
    export JAX_COMPILATION_CACHE_DIR=~/.cache/flock-zorch/jax-<frx version>
    PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" \\
        taskset -c 0-15 .venv/bin/python \\
        python/flock_zorch/testing/cpu_phase_split.py --log2 12 --mode split
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys
import tempfile
import time

import frx

frx.config.update("jax_enable_x64", True)

from frx import profiler  # noqa: E402

from flock_zorch.testing._bench_profile import (  # noqa: E402
    PHASES,
    BenchProver,
    constants_golden,
)
from flock_zorch.testing.blake3_ligerito_oracle_test import load  # noqa: E402

K_LOG = 14  # bench_worker.py's: 2^14 witness bits per compression, m = log2 + K_LOG

# The harness's untimed warm-up seed, and a distinct timed one — reusing the
# warm-up seed would measure a prove whose every buffer is already resident.
WARMUP_SEED = 0x00C0_FFEE_BEEF_D15C
TIMED_SEED = 0x5EED_0F_7A17

# Kernel classes, and how a traced HLO op is assigned to one.
#
# The rules are keyed on (module stem, op family) and every one of them was
# read off the dumped `after_optimizations` HLO, because the op NAME alone
# lies: `select_reduce-window_fusion` is the round-1 URM's
# binary_field_ghash[.,8,64] select + XOR-reduce inside `_round1_core`, and a
# u32[32] proof-of-work reduce inside `_open_jitted`. Same name, two classes,
# 470 ms apart. Anything not matched lands in `other`, which is reported as a
# residual rather than folded into a neighbour.
#
# The classes map onto the three repositories #323 lists levers in:
#   select_xor / ghash_mul -> prime-ir's binary-field lowering
#   transform / xor_fold   -> XLA:CPU codegen and parallelism
#   blake3                 -> the hash emitter on the CPU path
#   ffi / other            -> the host pipeline and dispatch

# (module stem, op family) pairs that must NOT follow the generic op rules.
_MODULE_OP_OVERRIDE = {
    # `grind_and_sample`'s PoW reduce — u32, not a field op.
    ("_open_jitted", "select_reduce-window_fusion"): "grind",
    ("_open_jitted", "wrapped_reduce"): "grind",
}

# Op-family substring -> class, checked in order; first hit wins.
_OP_CLASS = (
    ("ntt", "transform"),
    ("bit_reverse", "transform"),
    ("blake3", "blake3"),
    ("sha256", "blake3"),
    ("ffi_call", "ffi"),
    ("multiply", "ghash_mul"),
    ("reduce-window", "select_xor"),
)

# Modules whose whole body is one class, applied only after the op rules miss.
_MODULE_CLASS = {
    "_seg_xor_fold": "xor_fold",  # the lincheck segment fold: slice/pad/add
}

KERNEL_CLASSES = (
    "select_xor",
    "ghash_mul",
    "transform",
    "blake3",
    "xor_fold",
    "grind",
    "ffi",
    "other",
)


def _stem(module: str) -> str:
    """`jit__round1_core-<hash>` -> `_round1_core`, so a recompilation under a
    different shape lands in the same class."""
    return module.removeprefix("jit_").split("-")[0]


def _family(op: str) -> str:
    """`multiply_reduce-window_fusion.46` -> `multiply_reduce-window_fusion`.
    XLA numbers each instance of a fusion; the family is the unit worth
    ranking."""
    head, _, tail = op.rpartition(".")
    return head if head and tail.isdigit() else op


def classify(module: str, op: str) -> str:
    """Kernel class for one traced HLO op."""
    stem, fam = _stem(module), _family(op)
    override = _MODULE_OP_OVERRIDE.get((stem, fam))
    if override is not None:
        return override
    low = fam.lower()
    for needle, name in _OP_CLASS:
        if needle in low:
            return name
    return _MODULE_CLASS.get(stem, "other")


def _spread(xs: list[float]) -> str:
    """`min` with the run-to-run spread beside it. A single number from this
    harness is not a measurement — the repo's own rule."""
    lo, hi = min(xs), max(xs)
    med = statistics.median(xs)
    return (
        f"{lo:8.1f} med {med:8.1f} max {hi:8.1f}  spread {100 * (hi - lo) / lo:5.1f}%"
    )


def _prover(log2: int, golden: str | None):
    """`(BenchProver, startup_ms)` — startup is import-to-ready as the harness
    charges it: golden load plus the warm-up prove that compiles (or, on a warm
    cache, deserializes) every program the timed call runs."""
    t0 = time.perf_counter()
    m = log2 + K_LOG
    bp = BenchProver(load(golden or constants_golden(m)))
    bp.prove_bundle(WARMUP_SEED)
    return bp, (time.perf_counter() - t0) * 1e3


def mode_wall(bp: BenchProver, args) -> None:
    walls = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        bp.prove_bundle(TIMED_SEED)
        walls.append((time.perf_counter() - t0) * 1e3)
    n_hash = 1 << args.log2
    print(f"wall ms  (n={args.runs} in-process): {_spread(walls)}")
    print(f"comp/s at min: {n_hash * 1e3 / min(walls):.0f}  ({n_hash} compressions)")
    if args.json:
        print(JSON_MARK + json.dumps({"mode": "wall", "walls_ms": walls}))


def mode_split(bp: BenchProver, args) -> None:
    runs = []
    for _ in range(args.runs):
        times: dict[str, float] = {}
        t0 = time.perf_counter()
        bp.prove_bundle(TIMED_SEED, times)
        times["_wall"] = (time.perf_counter() - t0) * 1e3
        runs.append(times)

    print(f"{'phase':>10} {'min':>8} {'med':>9} {'max':>9} {'spread':>8} {'share':>7}")
    print("-" * 54)
    mins = {p: min(r[p] for r in runs) for p in PHASES}
    total = sum(mins.values())
    for p in PHASES:
        xs = [r[p] for r in runs]
        lo, hi = min(xs), max(xs)
        print(
            f"{p:>10} {lo:8.1f} {statistics.median(xs):9.1f} {hi:9.1f} "
            f"{100 * (hi - lo) / lo:7.1f}% {100 * mins[p] / total:6.1f}%"
        )
    walls = [r["_wall"] for r in runs]
    print("-" * 54)
    print(f"{'sum':>10} {total:8.1f}")
    print(f"{'wall':>10} {min(walls):8.1f}   (barriered; n={args.runs} in-process)")
    if args.json:
        print(JSON_MARK + json.dumps({"mode": "split", "runs": runs}))


def _phase_windows(planes) -> list[tuple[str, int, int]]:
    """`[(phase, start_ns, end_ns)]` from the `TraceAnnotation`s `prove_bundle`
    emits around each phase. Op events and annotations share one clock, so
    bucketing an op by its start timestamp is exact — no module->phase map to
    get wrong."""
    wins = []
    for plane in planes:
        for line in plane.lines:
            for e in line.events:
                if e.name in PHASES:
                    wins.append((e.name, e.start_ns, e.end_ns))
    return sorted(wins, key=lambda w: w[1])


def _phase_of(wins, ts: int) -> str:
    for name, lo, hi in wins:
        if lo <= ts < hi:
            return name
    return "outside"


def mode_ops(bp: BenchProver, args) -> None:
    d = tempfile.mkdtemp(prefix="cpu-phase-split-")
    times: dict[str, float] = {}
    t0 = time.perf_counter()
    with profiler.trace(d):
        bp.prove_bundle(TIMED_SEED, times)
    traced_ms = (time.perf_counter() - t0) * 1e3

    files = glob.glob(os.path.join(d, "**", "*.xplane.pb"), recursive=True)
    if not files:
        sys.exit(f"no xplane under {d} — is --xla_cpu_enable_xprof_traceme=true set?")

    planes = [pl for f in files for pl in profiler.ProfileData.from_file(f).planes]
    wins = _phase_windows(planes)
    if len(wins) != len(PHASES):
        sys.exit(
            f"expected one annotation per phase, saw {len(wins)}: "
            f"{[w[0] for w in wins]} — trace is not attributable"
        )

    by_class: dict[str, float] = collections.defaultdict(float)
    by_cell: dict[tuple[str, str], float] = collections.defaultdict(float)
    by_op: dict[tuple[str, str, str], list] = collections.defaultdict(lambda: [0.0, 0])
    missing = n_ev = 0
    for plane in planes:
        for line in plane.lines:
            for e in line.events:
                stats = dict(e.stats)
                op = stats.get("hlo_op")
                if op is None:
                    continue
                module = stats.get("hlo_module")
                if module is None:
                    missing += 1
                    continue
                n_ev += 1
                ms = e.duration_ns / 1e6
                cls = classify(str(module), str(op))
                by_class[cls] += ms
                by_cell[(_phase_of(wins, e.start_ns), cls)] += ms
                slot = by_op[(cls, str(module), _family(str(op)))]
                slot[0] += ms
                slot[1] += 1

    # A profiler that reports an unknown field as empty turns into a silent
    # zero three tables later. Refuse rather than round it away.
    if missing:
        sys.exit(f"{missing} traced ops carry no hlo_module — classification unsound")
    if not n_ev:
        sys.exit("trace carried no hlo_op events — traceme flag not in XLA_FLAGS?")

    busy = sum(by_class.values())
    barriered = sum(times[p] for p in PHASES)
    print(
        f"traced wall {traced_ms:.1f} ms | barriered sum {barriered:.1f} ms | "
        f"op busy {busy:.1f} ms | {n_ev} op events"
    )
    print(
        "  (tracing inflates the wall; op busy is the attributable quantity — "
        "compare it against a clean `--mode wall` run, never the traced wall)\n"
    )

    used = [c for c in KERNEL_CLASSES if by_class.get(c)]
    print(f"{'phase':>10} " + " ".join(f"{c:>10}" for c in used) + f" {'total':>9}")
    print("-" * (11 + 11 * len(used) + 10))
    for ph in (*PHASES, "outside"):
        row = [by_cell.get((ph, c), 0.0) for c in used]
        if sum(row) < 0.05:
            continue
        print(f"{ph:>10} " + " ".join(f"{v:10.1f}" for v in row) + f" {sum(row):9.1f}")
    print("-" * (11 + 11 * len(used) + 10))
    print(
        f"{'total':>10} "
        + " ".join(f"{by_class[c]:10.1f}" for c in used)
        + f" {busy:9.1f}"
    )
    print(
        f"{'share':>10} " + " ".join(f"{100 * by_class[c] / busy:9.1f}%" for c in used)
    )

    print(f"\ntop {args.top} op families")
    print(f"{'class':>12} {'ms':>9} {'n':>6}  module / family")
    print("-" * 74)
    for (cls, module, fam), (ms, n) in sorted(by_op.items(), key=lambda kv: -kv[1][0])[
        : args.top
    ]:
        print(f"{cls:>12} {ms:9.1f} {n:6}  {module} / {fam}")
    if args.json:
        print(
            JSON_MARK
            + json.dumps(
                {
                    "mode": "ops",
                    "traced_wall_ms": traced_ms,
                    "busy_ms": busy,
                    "phase_ms": times,
                    "by_class": dict(by_class),
                    "by_phase_class": {f"{p}|{c}": v for (p, c), v in by_cell.items()},
                    "by_op": {f"{c}|{m}|{o}": v for (c, m, o), v in by_op.items()},
                }
            )
        )


JSON_MARK = "##cpu-split-json## "

MODES = {"wall": mode_wall, "split": mode_split, "ops": mode_ops}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--log2", type=int, default=12, help="harness log2; m = log2 + 14")
    ap.add_argument("--golden", help="override the golden filename under artifacts/")
    ap.add_argument("--mode", choices=sorted(MODES), default="split")
    ap.add_argument("--runs", type=int, default=5, help="timed iterations in-process")
    ap.add_argument("--top", type=int, default=25, help="ops mode: rows to print")
    ap.add_argument("--json", action="store_true", help=f"also emit a {JSON_MARK} line")
    args = ap.parse_args()

    device = frx.devices()[0]
    if device.platform != "cpu":
        print(
            f"REFUSING: device is {device.platform}, not cpu. This harness reports "
            "the CPU tier; export FRX_PLATFORMS=cpu.",
            file=sys.stderr,
        )
        return 2

    bp, startup_ms = _prover(args.log2, args.golden)
    print(f"device {device} | m={args.log2 + K_LOG} | {1 << args.log2} compressions")
    print(f"startup (golden load + warm-up prove): {startup_ms / 1e3:.1f} s\n")
    MODES[args.mode](bp, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
