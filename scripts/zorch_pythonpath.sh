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

# Resolve the external repo root layout-independently as <output_base>/<workspace_root>.
# cquery materializes @zorch as a side effect, so no separate `bazel fetch` is needed;
# `label.workspace_root` avoids parsing a display string or guessing the tree depth.
base=$(bazel info output_base 2>/dev/null) || true
wsroot=$(bazel cquery @zorch//zorch:byte_transcript --output=starlark \
    --starlark:expr='target.label.workspace_root' 2>/dev/null | head -1) || true
[ -n "$base" ] && [ -n "$wsroot" ] || { echo "could not resolve @zorch path via bazel" >&2; exit 1; }
echo "$base/$wsroot"
