#!/usr/bin/env bash
# snark.fast harness worker entry — the env shim. The harness spawns its
# worker with env_clear() (only RAYON_NUM_THREADS and TMPDIR survive), so
# everything the GPU prover needs is restored here before exec'ing the
# python worker. Point the harness at THIS script (its WORKER positional):
#
#   cargo run --release -p flock-benchmark-harness -- \
#     <flock-zorch>/scripts/bench_worker.sh SCRATCH SCORE SUMMARY \
#     LOG2 THREADS WARMUP_RUNS RUNS
set -euo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin"
HOME="$(getent passwd "$(id -u)" | cut -d: -f6)"
export HOME
export PATH="$HOME/.local/cuda13/bin:$PATH" # ptxas (the CUDA 13 toolchain)

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export FRX_PLATFORMS=cuda,cpu
# The harness SIGKILLs the previous trial's worker right before spawning the
# next one, and the driver frees a killed process's VRAM asynchronously — a
# fresh worker that preallocates 75% of the card races that free and OOMs.
# On-demand allocation makes the overlap window cost only what is live.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)"
export PYTHONPATH

# Per-wheel XLA compile cache, deliberately OUTSIDE TMPDIR: the harness wipes
# its scratch (= TMPDIR) between trials, and a fresh worker per trial must
# absorb the multi-minute m32 compile inside the 300 s readiness budget — so
# warm trials MUST hit this cache. Keyed by the frx wheel version because
# shared caches across toolchains have served wrong executables before
# (docs/measurement.md); one dir per wheel is the standing rule.
frx_ver="$(.venv/bin/python -c 'import frx; print(frx.__version__)')"
export JAX_COMPILATION_CACHE_DIR="${FLOCK_ZORCH_JAX_CACHE:-$HOME/.cache/flock-zorch}/jax-$frx_ver"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# Cache EVERYTHING, including XLA's per-fusion autotune results and kernel
# cache: the readiness budget is spent on first-call compile work, and the
# default 1 s floor left most of it uncached (5 entries — ready took 406 s
# against the 300 s budget; the timed prove itself is seconds).
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all

exec .venv/bin/python python/flock_zorch/testing/bench_worker.py "$@"
