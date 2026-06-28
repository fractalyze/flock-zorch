#!/usr/bin/env bash
# Regenerate the byte-identity golden fixtures from the pinned flock submodule
# (third_party/flock). Each dump_<x> writes artifacts/<x>_golden.bin with its default
# size; the matching python/flock_zorch/testing/<x>_oracle_test.py reads it back.
#
#   scripts/dump_goldens.sh [core|all]      (default: core)
#     core = every layer + the identity e2e prover (fast; small fixtures)
#     all  = core + the real hash-circuit full provers (slow; blake3_golden ~118 MB)
#
# Note: a few gates sweep configs via their own runner rather than a single default
# golden — PCS commit (python/flock_zorch/testing/run_commit_gates.sh) and the
# multi-(m,k_log,k_skip) lincheck gate regenerate their own per-config goldens.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$HOME/.cargo/env" 2>/dev/null || true

[ -d third_party/flock/crates ] || { echo "third_party/flock missing — run: git submodule update --init"; exit 1; }
cargo build --release --examples >/dev/null

CORE=(field_mul ntt sha256 merkle merkle_multi commit sumcheck zerocheck challenger
      gf8_urm fri_fold row_batch ring_switch basefold pcs_open ligerito lincheck
      e2e chain_shift)
HEAVY=(keccak keccak_ligerito keccak_chain keccak3_ligerito sha2 sha2_ligerito blake3 blake3_ligerito)

mkdir -p artifacts
dump() { echo "  dump_$1"; "./target/release/examples/dump_$1"; }

for d in "${CORE[@]}"; do dump "$d"; done
if [ "${1:-core}" = all ]; then
  echo "-- heavy real hash-circuit goldens (slow) --"
  for d in "${HEAVY[@]}"; do dump "$d"; done
fi

echo "goldens in artifacts/ ($(ls artifacts/*.bin 2>/dev/null | wc -l) files)"
