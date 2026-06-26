#!/usr/bin/env bash
# Reproducible PCS-commit byte-identity + 10x gate across multiple (m, rate, batch)
# configurations. Regenerates each golden from unmodified flock, runs the jax gate,
# and tees a committed log to artifacts/commit_gate_results.txt — so the multi-size
# claim is backed by an artifact, not just a transcript.
#
# Run from the repo root:
#   python/flock_zorch/testing/run_commit_gates.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"
VENV="${VENV:-/home/jooman/fractalyze/zorch/.venv/bin/python}"
export PATH="$HOME/.local/cuda13/bin:$PATH"
LOG="artifacts/commit_gate_results.txt"

source "$HOME/.cargo/env" 2>/dev/null || true
cargo build --release --example dump_commit --example bench_commit_cpu >/dev/null 2>&1

# (m, log_inv_rate, log_batch_size): cover both RS rates (1/2, 1/4) and several
# interleave widths + witness sizes.
CONFIGS=("20 1 5" "24 1 5" "26 1 5" "22 2 1" "22 1 3" "18 2 4")

{
  echo "# flock-zorch PCS commit gate — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# CPU: unmodified flock (target-cpu=native, thin-LTO). GPU: flock-zorch + clmad."
  echo
} > "$LOG"

fail=0
for cfg in "${CONFIGS[@]}"; do
  read -r m lir lbs <<< "$cfg"
  ./target/release/examples/dump_commit "$m" "$lir" "$lbs" artifacts/commit_golden.bin >/dev/null
  out=$(JAX_PLATFORMS=cuda PYTHONPATH=python "$VENV" \
        python/flock_zorch/testing/commit_oracle_test.py 2>&1 || true)
  line=$(echo "$out" | grep -E "byte-identity|encode|GATE" || echo "NO OUTPUT")
  echo "=== m=$m rate=1/2^$lir batch=2^$lbs ===" | tee -a "$LOG"
  echo "$line" | tee -a "$LOG"
  echo | tee -a "$LOG"
  echo "$out" | grep -q "GATE PASS" || fail=1
done

if [ "$fail" -eq 0 ]; then
  echo "ALL CONFIGS: byte-identical + >=10x  PASS" | tee -a "$LOG"
else
  echo "SOME CONFIGS FAILED" | tee -a "$LOG"; exit 1
fi
