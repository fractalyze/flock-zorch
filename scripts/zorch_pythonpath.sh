#!/usr/bin/env bash
# PYTHONPATH entry for the git_override'd @zorch — for venv/heavy gates that aren't
# bazel targets:  PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" .venv/bin/python ...
#
# Asking bazel costs two client round-trips, which the snark.fast harness pays
# once per trial: `bench_worker_common.sh` calls this in every worker it spawns,
# inside the readiness budget. The answer is memoised under
# $FLOCK_ZORCH_JAX_CACHE.
#
# The memo is only as good as its key, and a wrong hit hands the worker a
# DIFFERENT zorch than the pin names — a substitution `docs/measurement.md`
# already warns costs a whole measurement. So the key covers every input to
# bazel's answer: this checkout's path (worktrees share the cache dir and do not
# share an output base), MODULE.bazel and its lock (the git_override pin), and
# .bazelrc plus the .bazelrc.user it try-imports (where an
# `--override_module=zorch=...` substitutes a local working copy). Set
# FLOCK_ZORCH_ZORCH_PYTHONPATH_NOCACHE=1 to bypass the memo entirely.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

cache_dir="${FLOCK_ZORCH_JAX_CACHE:-$HOME/.cache/flock-zorch}"
# `cat` on a missing .bazelrc.user contributes nothing, which is the right key
# for "no override present" — and adding the file later changes the key.
key=$({ cat MODULE.bazel MODULE.bazel.lock .bazelrc .bazelrc.user 2>/dev/null; pwd; } \
    | sha256sum | cut -d' ' -f1)
cache="$cache_dir/zorch-pythonpath-$key"

if [ -z "${FLOCK_ZORCH_ZORCH_PYTHONPATH_NOCACHE:-}" ] && [ -r "$cache" ]; then
    cached=$(cat "$cache")
    # A file inside the tree, not just the directory: an expunged output base
    # can leave the parent behind, and a PYTHONPATH entry that resolves to an
    # empty directory fails later, as an import error in the prover.
    if [ -f "$cached/zorch/__init__.py" ]; then
        echo "$cached"
        exit 0
    fi
fi

# <output_base>/<workspace_root>; cquery materializes @zorch, `|| true` keeps the check reachable.
base=$(bazel info output_base 2>/dev/null) || true
wsroot=$(bazel cquery @zorch//zorch:byte_transcript --output=starlark \
    --starlark:expr='target.label.workspace_root' 2>/dev/null | head -1) || true
[ -n "$base" ] && [ -n "$wsroot" ] || { echo "could not resolve @zorch path via bazel" >&2; exit 1; }

# Write through a temp so a reader never sees a half-written path; a cache that
# cannot be written is not an error, only the slow path again.
if mkdir -p "$cache_dir" 2>/dev/null; then
    tmp="$cache.$$"
    printf '%s\n' "$base/$wsroot" >"$tmp" 2>/dev/null && mv -f "$tmp" "$cache" 2>/dev/null || rm -f "$tmp"
fi
echo "$base/$wsroot"
