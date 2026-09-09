#!/usr/bin/env bash
# snark.fast harness worker entry — CPU tier (#322). Same worker and same
# harness contract as bench_worker.sh; the only difference is that the prover
# runs entirely on CPU, so the FRX CPU tier can be timed against the x86
# leaderboard on a box with no GPU (or with the GPU deliberately out of play).
#
#   cargo run --release -p flock-benchmark-harness -- \
#     <flock-zorch>/scripts/bench_worker_cpu.sh SCRATCH SCORE SUMMARY \
#     LOG2 THREADS WARMUP_RUNS RUNS
set -euo pipefail

export FRX_PLATFORMS=cpu
source "$(dirname "${BASH_SOURCE[0]}")/bench_worker_common.sh"
