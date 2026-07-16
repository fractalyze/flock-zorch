#!/usr/bin/env bash
# Regenerate the byte-identity golden fixtures from the pinned flock (the flock-core
# / flock-prover git rev dep in Cargo.toml). Each dump_<x> writes artifacts/<x>_golden.bin
# with its default size; the matching python/flock_zorch/testing/<x>_oracle_test.py reads it back.
#
#   scripts/dump_goldens.sh [core|all]      (default: core)
#     core = every layer + the identity e2e prover (fast; small fixtures)
#     all  = core + the real hash-circuit full provers (slow; blake3_golden ~118 MB)
#
# Note: the multi-(m,k_log,k_skip) lincheck gate sweeps configs via its own runner
# rather than a single default golden — it regenerates its own per-config goldens.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$HOME/.cargo/env" 2>/dev/null || true

cargo build --release --examples >/dev/null   # fetches flock (git rev dep) + builds the dumpers

CORE=(ntt sha256 merkle merkle_multi sumcheck zerocheck challenger
      gf8_urm fri_fold ring_switch ligerito lincheck e2e_ligerito chain_shift)
HEAVY=(keccak_ligerito keccak_chain keccak3_ligerito sha2_ligerito blake3_ligerito)

mkdir -p artifacts
dump() { echo "  dump_$1"; "./target/release/examples/dump_$1"; }

for d in "${CORE[@]}"; do dump "$d"; done

if [ "${1:-core}" = all ]; then
  echo "-- heavy real hash-circuit goldens (slow) --"
  for d in "${HEAVY[@]}"; do dump "$d"; done
fi

echo "goldens in artifacts/ ($(ls artifacts/*.bin 2>/dev/null | wc -l) files)"
