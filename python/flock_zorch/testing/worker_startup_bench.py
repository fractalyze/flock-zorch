# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Where a harness worker's readiness wall goes, and what a ranked run costs.

The snark.fast harness spawns a fresh worker per trial and polls for its ready
file before it starts the clock, so readiness is repaid once per trial while
never entering compressions/s (`docs/measurement.md`). This drives the real
entry script the way the harness does — `env_clear()` but for
`RAYON_NUM_THREADS` and `TMPDIR`, ready file, seed on stdin, proof rename — and
attributes the readiness wall to the phases `bench_worker.py` marks, min /
median / spread over N fresh workers.

    python/flock_zorch/testing/worker_startup_bench.py --log2 12 --runs 5

`--log2 12` is m26 (m = log2 + 14). Run it on a quiet box and report the
machine with the numbers: the phases are wall-clock, so a competing build
inflates all of them.

`--programs` breaks the warm-up phase out per XLA program from
`JAX_LOG_COMPILES`, and cross-checks each against the compilation cache **by
name**. Timing alone cannot tell a slow cache hit from an entry the cache
refused to write, so a program with no `jit_<name>-*-cache` file is flagged
rather than reported as a load — the distinction `docs/measurement.md` insists
on, and the one any claim about deserialization cost rests on.

Teardown is deliberately absent. The harness `kill`s the worker the moment it
captures the proof (`benchmark-tools/harness/src/main.rs::run_trial`), so
interpreter and XLA teardown are off its critical path however long they take,
and timing a hand-run worker to process exit charges them anyway — the trap
`docs/measurement.md` names.
"""

from __future__ import annotations

import argparse
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# `bench_worker.py`'s marks, in execution order; each row of the report is the
# span from the previous mark. The first span runs from this process's own
# pre-spawn timestamp, so `launch` is the entry script's env shim plus
# interpreter boot — work the worker itself cannot time.
PHASES = ("launch", "imports", "golden", "prover", "warmup", "ready")

# The seed the harness would write; any u64 does, the prove is seed-generic.
SEED = 0xD15C_BEEF_C0FF_EE00

_COMPILE_RE = re.compile(r"Finished XLA compilation of (\S+) in ([0-9.]+) sec")
# The name inside `jit(...)`, which is also the `jit_<name>-<hash>-cache` stem.
_JIT_NAME_RE = re.compile(r"^jit\((.+)\)$")

# The harness's own STARTUP_TIMEOUT. A worker this bench tolerates but the
# harness would not is a worker the harness gets no trial out of, so measuring
# it under a looser budget would report a readiness nobody can spend.
READY_TIMEOUT_S = 300.0


def _run_once(
    worker: Path, log2: int, threads: int, programs: bool
) -> tuple[dict, dict, str]:
    """One fresh worker through the harness's own contract.

    Returns `(spans, compiles, cache_dir)`: seconds per phase, seconds per XLA
    program (empty unless `programs`), and the compilation cache the worker
    chose, so the caller can check the programs against it by name. Raises if
    the worker never reaches readiness.
    """
    scratch = Path(tempfile.mkdtemp(prefix="flock-startup-"))
    try:
        ready, proof = scratch / "run.ready", scratch / "run.proof"
        timeline = scratch / "timeline.tsv"
        env = {
            "RAYON_NUM_THREADS": str(threads),
            "TMPDIR": str(scratch),
            "FLOCK_ZORCH_WORKER_TIMELINE": str(timeline),
        }
        if programs:
            env["JAX_LOG_COMPILES"] = "1"
        # stderr to a FILE, never a pipe: nobody drains a pipe while this waits
        # on the ready file, and the worker writes hundreds of KB before it —
        # the compile log plus two `cpu_aot_loader` feature dumps per cache hit,
        # each a few KB. A 64 KB pipe buffer fills and the worker blocks on
        # write, which reads as a startup that never finishes.
        log = scratch / "worker.err"
        spawn = time.time()
        with log.open("wb") as err_file:
            child = subprocess.Popen(
                [str(worker), str(log2), str(ready), str(proof)],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=err_file,
            )
            deadline = spawn + READY_TIMEOUT_S
            while not ready.is_file():
                if child.poll() is not None:
                    raise RuntimeError(
                        f"worker exited before readiness with status {child.returncode}"
                    )
                if time.time() >= deadline:
                    child.kill()
                    raise RuntimeError(
                        f"worker missed readiness in {READY_TIMEOUT_S:.0f} s"
                    )
                time.sleep(0.02)
            child.communicate(input=f"{SEED}\n".encode())
        if child.returncode != 0:
            raise RuntimeError(f"worker exited {child.returncode}")

        marks = _parse_timeline(timeline)
        spans, prev = {}, spawn
        for phase in PHASES:
            spans[phase] = float(marks[phase]) - prev
            prev = float(marks[phase])
        compiles = _parse_compiles(log.read_text(errors="replace")) if programs else {}
        return spans, compiles, marks.get("cache_dir", "")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _parse_timeline(path: Path) -> dict[str, str]:
    """`bench_worker`'s timeline: the phase marks as epoch seconds, plus the
    `cache_dir` row. Values stay strings — one of them is a path."""
    rows = (line.split("\t", 1) for line in path.read_text().splitlines() if line)
    return {key: value for key, value in rows}


def _parse_compiles(stderr: str) -> dict[str, float]:
    """Seconds per program name, summed over its (differently shaped) entries."""
    per_program: dict[str, float] = {}
    for name, seconds in _COMPILE_RE.findall(stderr):
        per_program[name] = per_program.get(name, 0.0) + float(seconds)
    return per_program


def _cache_misses(programs: set[str], cache_dir: str) -> set[str]:
    """Programs with no cache entry of their own name.

    A miss recompiles in full in every worker, and its seconds are compilation
    rather than deserialization — the two are indistinguishable by timing, so
    the directory listing is the discriminator (`docs/measurement.md`).
    """
    if not cache_dir or not Path(cache_dir).is_dir():
        return set()
    stems = {entry.name.rsplit("-", 2)[0] for entry in Path(cache_dir).iterdir()}
    missing = set()
    for program in programs:
        name = _JIT_NAME_RE.match(program)
        if name and f"jit_{name.group(1)}" not in stems:
            missing.add(program)
    return missing


def _report(
    runs: list[dict], label: str, unit_total: str, flag: set[str] | None = None
) -> None:
    """Print `min / median / spread` per key, widest first, then the total.

    Keys are the union over runs, not the first run's: a program only some runs
    compiled still belongs in a table whose total counts it.
    """
    flag = flag or set()
    keys = sorted(
        {k for r in runs for k in r},
        key=lambda k: -min(r.get(k, 0.0) for r in runs),
    )
    width = max(len(k) for k in list(keys) + [unit_total, label]) + 2

    def row(name: str, seconds: list[float], mark: str = "") -> None:
        lo = min(seconds)
        print(
            f"{name + mark:{width}s}  {lo * 1e3:9.1f}  "
            f"{statistics.median(seconds) * 1e3:9.1f}  {(max(seconds) - lo) * 1e3:9.1f}"
        )

    print(f"{label:{width}s}  {'min ms':>9}  {'median ms':>9}  {'spread ms':>9}")
    for key in keys:
        row(key, [r.get(key, 0.0) for r in runs], " *" if key in flag else "")
    row(unit_total, [sum(r.values()) for r in runs])
    if flag:
        print("* no cache entry of this name — recompiled, not deserialized")


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log2", type=int, default=12, help="harness log2 (m = log2 + 14)")
    ap.add_argument("--runs", type=int, default=5, help="fresh workers to spawn")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument(
        "--worker",
        type=Path,
        default=repo / "scripts" / "bench_worker_cpu.sh",
        help="harness entry script (the CPU tier by default)",
    )
    ap.add_argument(
        "--programs",
        action="store_true",
        help="also break the XLA cache load out per program (JAX_LOG_COMPILES)",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=120,
        help="ranked trial count to project a run's wall time from",
    )
    args = ap.parse_args(argv)

    runs: list[dict] = []
    compiles: list[dict] = []
    cache_dir = ""
    for i in range(args.runs):
        spans, per_program, cache_dir = _run_once(
            args.worker.resolve(), args.log2, args.threads, args.programs
        )
        runs.append(spans)
        if per_program:
            compiles.append(per_program)
        print(
            f"run {i + 1}/{args.runs}: readiness {sum(spans.values()):.2f} s",
            file=sys.stderr,
        )

    m = args.log2 + 14
    print(
        f"\n=== readiness, m{m}, {args.runs} fresh workers, {args.threads} threads ==="
    )
    _report(runs, "phase", "readiness")
    if compiles:
        print("\n=== XLA cache load per program (of the warmup phase) ===")
        seen = {k for r in compiles for k in r}
        _report(compiles, "program", "all programs", _cache_misses(seen, cache_dir))
    readiness = min(sum(r.values()) for r in runs)
    print(
        f"\nranked {args.trials}-trial run pays readiness {args.trials} times: "
        f"{readiness * args.trials / 60:.1f} min before a single prove."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
