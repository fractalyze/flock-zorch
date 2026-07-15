#!/usr/bin/env bash
# flock-zorch bootstrap — get a fresh GPU box to a runnable (gates + benches) state.
#
# Idempotent; safe to re-run. From the repo root:
#   scripts/setup.sh
#
# Overridable via env: PYTHON_BIN (default python3.11), VENV_DIR (default ./.venv).
# See README.md for prerequisites, the optional clmad GPU acceleration, the full
# golden set, and how to run the whole gate suite + benchmarks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYPI_INDEX="https://fractalyze.github.io/pypi/simple/"

log()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

log "0. preflight"
have git    || { echo "git not found"; exit 1; }
have cargo  || { echo "cargo (Rust) not found — install rustup (flock-core is edition 2024)"; exit 1; }
have "$PYTHON_BIN" || { echo "$PYTHON_BIN not found — need Python 3.11"; exit 1; }
source "$HOME/.cargo/env" 2>/dev/null || true
if have nvidia-smi; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
else
  echo "WARN: no nvidia-smi — the GPU gates/benches need a CUDA GPU (RTX 5090 / sm_120 reference)"
fi

log "1. deps — no submodules: flock is a cargo git rev dep (fetched by the build in step 3); zorch is a bazel git_override"

log "2. python venv -> $VENV_DIR (frx jax-fork + frxlib + zk_dtypes + CUDA PJRT from the fractalyze index)"
[ -d "$VENV_DIR" ] || "$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install -q --upgrade pip
"$VENV_DIR/bin/pip" install -q -r requirements.in --extra-index-url "$PYPI_INDEX"
"$VENV_DIR/bin/python" -c 'import frx; print("  frx", frx.__version__)'

log "3. build Rust — fetches flock (git rev dep), builds golden dumpers + CPU benches"
cargo build --release             # lib (rlib)
cargo build --release --examples  # dump_* (goldens) + bench_*_cpu (apple-to-apple CPU baselines)

log "4. clmad GPU fast path (compiler-emitted by the frx wheel; needs a CUDA 13.3 ptxas on PATH at runtime — nothing to build)"
if [ -x "${PTXAS:-$HOME/.local/cuda13/bin/ptxas}" ]; then
  echo "  CUDA-13.x ptxas found — put ~/.local/cuda13/bin on PATH and the pinned frx"
  echo "  wheel emits hardware clmad for the GF(2^128) multiply."
else
  echo "  no CUDA-13.x ptxas — gates/benches use the software binary_field_ghash multiply"
  echo "  (byte-identical, slower). See README.md 'Setup' (clmad fast path)."
fi

log "5. regenerate the core golden fixtures from the pinned flock"
scripts/dump_goldens.sh core

log "6. smoke gates — byte-identity end-to-end (CPU; bazel manages the pip deps + the git_override'd zorch)"
bazel test //python:sumcheck_oracle_test //python:e2e_oracle_test

log "DONE — box is set up and byte-identity is green."
cat <<EOF

To run more:
  # the core byte-identity gates (CPU, bazel-managed, zorch via git_override):
  bazel test //python:all

  # full golden set incl. the heavy real hash circuits (keccak/sha2/blake3, blake3 ~118MB):
  scripts/dump_goldens.sh all
  # a heavy/GPU gate (not a bazel target) — resolve zorch from the same git_override:
  export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
  PYTHONPATH="python:\$(scripts/zorch_pythonpath.sh)" $VENV_DIR/bin/python \\
      python/flock_zorch/testing/<name>_oracle_test.py
  # benchmarks (run on an IDLE machine for an honest apple-to-apple CPU baseline):
  see README.md  ("Benchmark")
EOF
