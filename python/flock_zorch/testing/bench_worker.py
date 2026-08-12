# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""snark.fast harness worker — the GPU prover through the flock-challenge
benchmark window.

The harness (`benchmark-tools/harness`) spawns a FRESH worker per trial as
`<worker> <log2> <ready> <proof>` with a cleared env (only RAYON_NUM_THREADS
and TMPDIR survive), polls for the ready file (300 s budget), then writes a
decimal u64 seed to stdin; the trial clock stops when the proof file appears.
This module is the timed side of that contract: untimed warm-up prove at the
same log2 (the seed is traced, so it compiles every program the timed call
runs), ready file, seed → `BenchProver.prove_bundle` → write + atomic rename.
stdout/stderr are discarded by the harness.

Point the harness at `scripts/bench_worker.sh`, which restores the env
(PATH/PYTHONPATH/per-wheel JAX cache) before exec'ing this module — without
the cache a respawned worker pays the multi-minute m32 XLA compile against
the 300 s readiness budget.

The bundle bytes this emits are byte-gated against a fork-verified golden by
`bench_ligerito_oracle_test.py` (same `BenchProver`, same constants path).
"""

import os
import sys

import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch.testing._bench_profile import (  # noqa: E402
    BenchProver,
    constants_golden,
)
from flock_zorch.testing.blake3_ligerito_oracle_test import load  # noqa: E402

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

    bp = BenchProver(load(constants_golden(log2 + K_LOG)))
    bp.prove_bundle(WARMUP_SEED)  # untimed: compile + first-touch everything

    with open(ready_path, "wb") as f:
        f.write(b"ready\n")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
