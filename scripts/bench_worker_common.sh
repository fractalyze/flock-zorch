#!/usr/bin/env bash
# Shared body of the snark.fast harness worker entries — the env shim. The
# harness spawns its worker with env_clear() (only RAYON_NUM_THREADS and TMPDIR
# survive), so everything the prover needs is restored here before exec'ing the
# python worker. That cleared env is also why the platform list cannot be an
# override from the caller: each tier gets its own entry script, which sets
# FRX_PLATFORMS and then sources this file.
#
#   scripts/bench_worker.sh      cuda,cpu  — the GPU tier
#   scripts/bench_worker_cpu.sh  cpu       — the CPU tier (#322)
#
# Not executable on its own; point the harness at an entry script instead.
: "${FRX_PLATFORMS:?entry script must export FRX_PLATFORMS before sourcing}"

export PATH="/usr/local/bin:/usr/bin:/bin"
HOME="$(getent passwd "$(id -u)" | cut -d: -f6)"
export HOME
export PATH="$HOME/.local/cuda13/bin:$PATH" # ptxas (the CUDA 13 toolchain)

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# The harness SIGKILLs the previous trial's worker right before spawning the
# next one, and the driver frees a killed process's VRAM asynchronously — a
# fresh worker that preallocates 75% of the card races that free and OOMs.
# On-demand allocation makes the overlap window cost only what is live.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)"
export PYTHONPATH

# Per-wheel XLA compile cache, deliberately OUTSIDE TMPDIR: the harness wipes
# its scratch (= TMPDIR) between trials, and every fresh worker must absorb the
# whole XLA compile inside the harness's readiness budget, so warm trials have
# to hit this cache. Keyed by the frx wheel version: a cache key is only as
# good as what it covers, and a dir shared across toolchains can serve an
# executable built by a different one. The tiers share the dir, because JAX's
# own cache key already carries the backend and device kind, so a cpu-only and
# a cuda,cpu worker cannot read each other's entries.
frx_ver="$(.venv/bin/python -c 'import frx; print(frx.__version__)')"
export JAX_COMPILATION_CACHE_DIR="${FLOCK_ZORCH_JAX_CACHE:-$HOME/.cache/flock-zorch}/jax-$frx_ver"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# Cache EVERYTHING, including XLA's per-fusion autotune results and kernel
# cache. The readiness budget is spent almost entirely on first-call compile
# work, and the default minimum-compile-time floor leaves most of that
# uncached — few enough entries that a respawned worker recompiles past the
# budget and never reaches readiness.
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all

exec .venv/bin/python python/flock_zorch/testing/bench_worker.py "$@"
