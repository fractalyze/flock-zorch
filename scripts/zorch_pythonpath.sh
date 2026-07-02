#!/usr/bin/env bash
# PYTHONPATH entry for the git_override'd @zorch — for venv/heavy gates that aren't
# bazel targets:  PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" .venv/bin/python ...
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# <output_base>/<workspace_root>; cquery materializes @zorch, `|| true` keeps the check reachable.
base=$(bazel info output_base 2>/dev/null) || true
wsroot=$(bazel cquery @zorch//zorch:byte_transcript --output=starlark \
    --starlark:expr='target.label.workspace_root' 2>/dev/null | head -1) || true
[ -n "$base" ] && [ -n "$wsroot" ] || { echo "could not resolve @zorch path via bazel" >&2; exit 1; }
echo "$base/$wsroot"
