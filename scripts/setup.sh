#!/usr/bin/env bash
# flock-zorch bootstrap — get a fresh GPU box to a runnable (gates + benches) state.
#
# Idempotent; safe to re-run. From the repo root:
#   scripts/setup.sh
#
# Overridable via env: PYTHON_BIN (default python3.11), VENV_DIR (default ./.venv).
# See docs/SETUP.md for prerequisites, the optional clmad GPU acceleration, the full
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

log "1. submodules — flock + zorch pinned (see .gitmodules)"
git submodule update --init --recursive
git submodule status

log "2. python venv -> $VENV_DIR (jax fork + zk_dtypes + zkx CUDA PJRT from the fractalyze index)"
[ -d "$VENV_DIR" ] || "$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install -q --upgrade pip
"$VENV_DIR/bin/pip" install -q -r requirements.in --extra-index-url "$PYPI_INDEX"
"$VENV_DIR/bin/python" -c 'import jax; print("  jax", jax.__version__)'

log "3. build Rust against third_party/flock — cdylib (host SHA-NI Merkle FFI) + golden dumpers + CPU benches"
cargo build --release             # lib: rlib + cdylib
cargo build --release --examples  # dump_* (goldens) + bench_*_cpu (apple-to-apple CPU baselines)

log "4. clmad GPU FFI (OPTIONAL — needs CUDA 13.x ptxas + sm_120; gates fall back to software field.mul)"
if [ -x "${PTXAS:-$HOME/.local/cuda13/bin/ptxas}" ]; then
  echo "  ptxas found — building the clmad cubin + handler (best-effort)"
  VENV="$VENV_DIR/bin/python" bash optim/clmad/build_ffi.sh || echo "  WARN: clmad build failed; gates/benches use software field.mul (slower, byte-identical)"
else
  echo "  skip: no CUDA-13.x ptxas. clmad is the GPU fast path; without it everything still runs"
  echo "        (software field.mul, byte-identical). See optim/clmad/README.md + docs/SETUP.md."
fi

log "5. regenerate the core golden fixtures from the pinned flock"
scripts/dump_goldens.sh core

log "6. smoke gates — byte-identity end-to-end (low GPU-mem so it co-exists on a shared box)"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="python:third_party/zorch"
"$VENV_DIR/bin/python" python/flock_zorch/testing/field_oracle_test.py
"$VENV_DIR/bin/python" python/flock_zorch/testing/e2e_oracle_test.py

log "DONE — box is set up and byte-identity is green."
cat <<EOF

To run more, set the gate environment once:
  export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=python:third_party/zorch
  VENV=$VENV_DIR/bin/python

  # full golden set incl. the heavy real hash circuits (keccak/sha2/blake3, blake3 ~118MB):
  scripts/dump_goldens.sh all
  # any gate:
  \$VENV python/flock_zorch/testing/<name>_oracle_test.py
  # benchmarks (run on an IDLE machine for an honest apple-to-apple CPU baseline):
  see docs/SETUP.md  ("Benchmarks")
EOF
