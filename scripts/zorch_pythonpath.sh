#!/usr/bin/env bash
# Print the PYTHONPATH entry for the git_override'd @zorch (pinned in MODULE.bazel).
#
# The core byte-match gates run under `bazel test //python:all`. The heavy
# hash-circuit gates (blake3/keccak/sha2, hundreds-of-MB goldens) and the commit
# GPU perf gate are NOT bazel py_test targets; run them via the venv, pointing
# PYTHONPATH at the same zorch bazel resolves — no third_party/zorch submodule:
#
#   PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" .venv/bin/python \
#       python/flock_zorch/testing/blake3_oracle_test.py
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Materialize the repo, then resolve its root from a known target's BUILD file:
#   <output_base>/external/zorch+/zorch/BUILD.bazel  ->  <output_base>/external/zorch+
bazel fetch @zorch//zorch:byte_transcript >/dev/null 2>&1 || true
loc=$(bazel query --output=location "@zorch//zorch:byte_transcript" 2>/dev/null | head -1)
build_file="${loc%%:*}"
[ -n "$build_file" ] || { echo "could not resolve @zorch path via bazel" >&2; exit 1; }
dirname "$(dirname "$build_file")"
