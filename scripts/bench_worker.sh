#!/usr/bin/env bash
# snark.fast harness worker entry — GPU tier. Point the harness at THIS script
# (its WORKER positional):
#
#   cargo run --release -p flock-benchmark-harness -- \
#     <flock-zorch>/scripts/bench_worker.sh SCRATCH SCORE SUMMARY \
#     LOG2 THREADS WARMUP_RUNS RUNS
#
# The env shim itself lives in bench_worker_common.sh, shared with the CPU tier.
set -euo pipefail

export FRX_PLATFORMS=cuda,cpu # GPU prover + CPU SHA query chains
source "$(dirname "${BASH_SOURCE[0]}")/bench_worker_common.sh"
