# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""snark.fast harness worker — the prover through the flock-challenge
benchmark window, on whichever platform the entry script selected.

The harness (`benchmark-tools/harness`) spawns a FRESH worker per trial as
`<worker> <log2> <ready> <proof>` with a cleared env (only RAYON_NUM_THREADS
and TMPDIR survive), polls for the ready file (300 s budget), then writes a
decimal u64 seed to stdin; the trial clock stops when the proof file appears.
This module is the timed side of that contract: untimed warm-up prove at the
same log2 (the seed is traced, so it compiles every program the timed call
runs), ready file, seed → `BenchProver.prove_bundle` → write + atomic rename.
stdout/stderr are discarded by the harness.

Point the harness at an entry script, never at this module: `bench_worker.sh`
for the GPU tier or `bench_worker_cpu.sh` for the CPU one. Each exports its
`FRX_PLATFORMS` and sources `bench_worker_common.sh`, which restores the env
(PATH/PYTHONPATH/per-wheel JAX cache) before exec'ing this module. Without a
warm cache a respawned worker recompiles from scratch and misses the readiness
budget, so the harness never gets a trial out of it.

The bundle bytes this emits are byte-gated against a fork-verified golden by
`bench_ligerito_oracle_test.py` (same `BenchProver`, same constants path).

`FLOCK_ZORCH_WORKER_TIMELINE` names a file to write this startup's phase
timeline to; `worker_startup_bench.py` sets it to attribute the readiness wall.
The harness clears the worker's env, so on the ranked path it is one lookup
and every `_mark` is a no-op — the instrument cannot move the number it
measures.
"""

import os
import sys
import time

_TIMELINE = os.environ.get("FLOCK_ZORCH_WORKER_TIMELINE")
# `launch` is the first instant this module runs: the parent's spawn timestamp
# minus this one is the entry script's env shim plus interpreter boot, neither
# of which any in-process clock sees.
_MARKS = [("launch", time.time())]


def _mark(name: str) -> None:
    if _TIMELINE:
        _MARKS.append((name, time.time()))


import frx  # noqa: E402

frx.config.update("jax_enable_x64", True)

from flock_zorch.testing._bench_profile import (  # noqa: E402
    BenchProver,
    constants_golden,
)

# `_golden` rather than the oracle test that re-exports it: importing the gate
# pulls its whole dependency set (the challengers, the pcs oracle helpers) into
# every harness worker for one function.
from flock_zorch.testing._golden import latest_blake3_golden as load  # noqa: E402

_mark("imports")

# The reference worker's untimed warm-up seed (benchmark-tools/worker).
WARMUP_SEED = 0x00C0_FFEE_BEEF_D15C

K_LOG = 14  # 2^14 witness bits per compression: m = log2 + K_LOG


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: bench_worker.py LOG2 READY_PATH PROOF_PATH", file=sys.stderr)
        return 2
    log2, ready_path, proof_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    if not 8 <= log2 <= 20:
        print("harness worker contract: log2 in 8..=20", file=sys.stderr)
        return 2

    g = load(constants_golden(log2 + K_LOG))
    _mark("golden")
    bp = BenchProver(g)
    _mark("prover")
    bp.prove_bundle(WARMUP_SEED)  # untimed: compile + first-touch everything
    _mark("warmup")

    with open(ready_path, "wb") as f:
        f.write(b"ready\n")
    _mark("ready")
    line = sys.stdin.readline()
    if not line:
        print("missing seed on stdin", file=sys.stderr)
        return 1
    seed = int(line.strip())

    data = bp.prove_bundle(seed)
    tmp = proof_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.rename(tmp, proof_path)
    _mark("proof")
    _write_timeline()
    return 0


def _write_timeline() -> None:
    """Dump the marks as `<key>\t<value>` lines, plus the compilation cache the
    entry script chose — which is the worker's to report, since the shim derives
    it and a reader checking programs against the wrong directory would call
    every one of them a miss. Written after the rename, so it is outside both
    the harness's timed window and its readiness poll even when a timeline run
    leaves the variable set."""
    if not _TIMELINE:
        return
    with open(_TIMELINE, "w") as f:
        for name, t in _MARKS:
            f.write(f"{name}\t{t:.6f}\n")
        f.write(f"cache_dir\t{os.environ.get('JAX_COMPILATION_CACHE_DIR', '')}\n")


if __name__ == "__main__":
    sys.exit(main())
