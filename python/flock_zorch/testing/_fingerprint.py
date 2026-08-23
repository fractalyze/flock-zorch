# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The environment facts that decide whether two benchmark numbers compare.

[`docs/measurement.md`](../../../docs/measurement.md) is the prose version of
this module. Every field below exists because getting it wrong has silently
moved a published number, and in each case the run itself looked fine — that
is what makes the fingerprint worth collecting rather than the failure being
left to show up as a mystery regression.

The severity table is the load-bearing part. A `BLOCKING` field has a measured
effect on the wall big enough that a delta measured across a change in it says
nothing about the code: the toolchain gap is 5.5-16x, a stale
`--override_module` hid 35% at m32 and cost every wall measured under it
(#200 erratum), the async allocator inflates m32 ~14%. An `INFORMATIONAL`
field explains a run without invalidating a comparison, so it is recorded and
never blocks.

`source` is deliberately outside the table: the commit under test is the
independent variable of the whole exercise, so it changes between every pair
of runs worth comparing and can never count as drift.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, NamedTuple

from flock_zorch.testing._ptxas import (
    clmad_nvlink_verdict,
    clmad_ptxas_verdict,
    nvlink_version_text,
    ptxas_version_text,
)

# Bumped when a field's meaning changes, so a consumer reading an old record
# out of a check run knows whether it may compare it against a new one. Adding
# a field does not need a bump — a reader that does not know it ignores it.
SCHEMA_VERSION = 1

BLOCKING = "blocking"
INFORMATIONAL = "informational"

# Dotted paths whose change makes two measurements incomparable. Anything not
# listed is informational, so a field added to the record later fails open
# (recorded, never blocking) rather than silently rejecting every comparison.
_BLOCKING_PATHS = (
    "toolchain.ptxas",
    "toolchain.nvlink",
    "device.name",
    "device.compute_capability",
    "overrides",
    "pins.installed",
    "pins.declared",
    "runtime.XLA_PYTHON_CLIENT_ALLOCATOR",
    "runtime.XLA_PYTHON_CLIENT_PREALLOCATE",
    "runtime.XLA_FLAGS",
    "window",
)

# The environment variables that reach the measurement. `CUDA_DIR` is here
# because it is where XLA resolves ptxas from, `CUDA_ROOT` because reading it
# back is the classic way to *think* the toolchain is right (frx exports it at
# import time as a libdevice hint, so its value says nothing about what
# assembled the kernels — see `_ptxas.ptxas_version_text`).
_RUNTIME_VARS = (
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_FLAGS",
    "CUDA_DIR",
    "CUDA_ROOT",
    "FRX_PLATFORMS",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
)

# Distributions whose version decides what the prover actually runs. The
# lockstep rule between these and the zorch pin is in `requirements.in`; the
# fingerprint records both sides so a broken lockstep is visible in the record
# instead of having to be remembered.
_TRACKED_DISTS = (
    "frx",
    "frxlib",
    "frx-cuda12-plugin",
    "frx-cuda12-pjrt",
    "hash-frx",
    "zk-dtypes",
)

_RELEASE_RE = re.compile(r"release (\d+\.\d+)")
# A bazelrc line is a comment only when `#` opens it; bazel has no trailing
# comments, so a `#` anywhere else is part of the value.
_OVERRIDE_RE = re.compile(r"^[^#\n]*--override_module=([A-Za-z0-9_.-]+)=(\S+)", re.M)
_GIT_OVERRIDE_RE = re.compile(
    r"git_override\((.*?)\)",
    re.S,
)
_REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9_.-]+)==(\S+)", re.M)


class Drift(NamedTuple):
    """One field that differs between two fingerprints."""

    path: str
    before: Any
    after: Any
    severity: str

    def describe(self) -> str:
        return f"{self.path}: {self.before!r} -> {self.after!r}"


# ------------------------------------------------------------------ parsing
# Pure functions over file contents, so the accidents that motivate them are
# testable without a box that reproduces the accident.


def parse_override_modules(bazelrc_text: str) -> dict[str, str]:
    """Active `--override_module=<module>=<path>` entries in a bazelrc.

    An override silently substitutes the source being measured, which is why
    `.bazelrc.user` tells a human to comment it out before recording a number.
    A commented line is inert, so only lines with no leading `#` count.
    """
    return {m.group(1): m.group(2) for m in _OVERRIDE_RE.finditer(bazelrc_text)}


def parse_module_pins(module_bazel_text: str) -> dict[str, str]:
    """`{module_name: commit}` for every `git_override` in a MODULE.bazel."""
    pins = {}
    for block in _GIT_OVERRIDE_RE.finditer(module_bazel_text):
        body = block.group(1)
        name = re.search(r'module_name\s*=\s*"([^"]+)"', body)
        commit = re.search(r'commit\s*=\s*"([0-9a-f]+)"', body)
        if name and commit:
            pins[name.group(1)] = commit.group(1)
    return pins


def parse_requirement_pins(requirements_text: str) -> dict[str, str]:
    """`{distribution: version}` for the `==` pins in a requirements file."""
    return {
        m.group(1).lower().replace("_", "-"): m.group(2)
        for m in _REQUIREMENT_RE.finditer(requirements_text)
    }


def release(version_text: str | None) -> str | None:
    """`"13.3"` out of a CUDA tool's `--version` banner, or None."""
    m = _RELEASE_RE.search(version_text) if version_text else None
    return m.group(1) if m else None


def drift(before: dict[str, Any], after: dict[str, Any]) -> list[Drift]:
    """Every field that differs between two fingerprints, severity-classified.

    Answers the question the ledger exists for — "may I compare these two?" —
    rather than "did anything change", which is always yes. Callers act on the
    `BLOCKING` subset and print the rest as context.
    """
    out: list[Drift] = []
    for path, b, a in _walk(before, after, ""):
        out.append(Drift(path, b, a, _severity(path)))
    return out


def blocking(drifts: list[Drift]) -> list[Drift]:
    return [d for d in drifts if d.severity == BLOCKING]


def publish_refusal(fingerprint: dict[str, Any]) -> str | None:
    """Why this environment must not add a point to a trajectory, or None.

    Deliberately stricter than the bench's own pre-flight guard. That guard
    fails open on an unreadable probe so it never blocks a human, who can see
    what box they are standing on; an unattended run has no such reader, and a
    toolchain that cannot assemble clmad produces a number ~15x slow that
    reads as a regression in whatever merged last rather than as an error.

    So: a known-capped toolchain is refused, and so is one that could not be
    identified at all. A card too old for clmad still reports both versions,
    so it is published normally — the fallback IS the measurement there.
    """
    tool = fingerprint["toolchain"]
    for reason in (tool["clmad_ptxas_refusal"], tool["clmad_nvlink_refusal"]):
        if reason:
            return str(reason)
    missing = [t for t in ("ptxas", "nvlink") if not tool[t]]
    if missing:
        return (
            f"{' and '.join(missing)} could not be identified, so there is no "
            "evidence this run assembled the clmad fast path"
        )
    return None


def _severity(path: str) -> str:
    return (
        BLOCKING
        if any(path == p or path.startswith(p + ".") for p in _BLOCKING_PATHS)
        else INFORMATIONAL
    )


def _walk(before: Any, after: Any, prefix: str) -> list[tuple[str, Any, Any]]:
    """Leaf-level diff of two nested dicts, as `(dotted path, before, after)`."""
    if isinstance(before, dict) and isinstance(after, dict):
        out: list[tuple[str, Any, Any]] = []
        for key in sorted(set(before) | set(after)):
            child = f"{prefix}.{key}" if prefix else key
            out += _walk(before.get(key), after.get(key), child)
        return out
    return [] if before == after else [(prefix, before, after)]


# ---------------------------------------------------------------- collection


def _run(cmd: list[str], cwd: str | None = None) -> str | None:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=True, cwd=cwd
        ).stdout.strip()
    except Exception:
        return None


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _smi(query: str) -> list[str]:
    out = _run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"])
    return [ln.strip() for ln in out.splitlines() if ln.strip()] if out else []


def _device() -> dict[str, Any]:
    """Card identity, read without taking a CUDA context.

    Deliberately not `prove_phase_bench.gpu_provenance`: that module imports
    frx at module scope, and a pre-flight probe that takes a context would
    make itself one of the processes a contention gate is looking for.
    """
    rows = [r.split(", ") for r in _smi("gpu=name,compute_cap,driver_version")]
    if not rows:
        return {"name": None, "compute_capability": None, "driver": None}
    return {
        "name": rows[0][0],
        "compute_capability": rows[0][1],
        "driver": rows[0][2],
        "count": len(rows),
    }


def _toolchain(compute_capability: str | None) -> dict[str, Any]:
    ptxas, nvlink = ptxas_version_text(), nvlink_version_text()
    return {
        "ptxas": release(ptxas),
        "nvlink": release(nvlink),
        # The verdicts, not just the versions: a reader of the record should
        # not have to know the clmad floor to see that a run was capped.
        "clmad_ptxas_refusal": clmad_ptxas_verdict(ptxas, compute_capability),
        "clmad_nvlink_refusal": clmad_nvlink_verdict(nvlink, compute_capability),
    }


def _overrides(repo_root: str) -> dict[str, Any]:
    """Active module overrides, each against the pin it is standing in for.

    Recording the override's checkout sha next to the declared pin is the
    whole point: the #200 erratum was not "an override was set" but "the
    override was 18 commits behind what the repo declared", and only the pair
    shows that.
    """
    active = parse_override_modules(_read(os.path.join(repo_root, ".bazelrc.user")))
    pins = parse_module_pins(_read(os.path.join(repo_root, "MODULE.bazel")))
    out: dict[str, Any] = {}
    for module, path in active.items():
        head = _run(["git", "rev-parse", "HEAD"], cwd=path)
        declared = pins.get(module)
        out[module] = {
            "path": path,
            "head": head,
            "declared_pin": declared,
            "matches_pin": bool(head and declared and head == declared),
        }
    return out


def _installed_versions(python: str) -> dict[str, str | None]:
    """Versions from the venv's installed metadata, not from the lock file.

    Reads distribution metadata rather than importing frx: the import is slow,
    and a hand-built wheel left in the venv (one of the recorded accidents) is
    exactly the case where the lock file and reality disagree, so the question
    has to be put to the venv.
    """
    # A loop rather than one expression so a distribution missing from the
    # venv reports None instead of aborting the whole probe.
    program = (
        "import importlib.metadata as m, json\n"
        f"DISTS = {list(_TRACKED_DISTS)!r}\n"
        "out = {}\n"
        "for d in DISTS:\n"
        "    try:\n"
        "        out[d] = m.version(d)\n"
        "    except Exception:\n"
        "        out[d] = None\n"
        "print(json.dumps(out))\n"
    )
    raw = _run([python, "-c", program])
    if not raw:
        return {d: None for d in _TRACKED_DISTS}
    try:
        return dict(json.loads(raw))
    except ValueError:
        return {d: None for d in _TRACKED_DISTS}


def _pins(repo_root: str, python: str) -> dict[str, Any]:
    declared = parse_requirement_pins(_read(os.path.join(repo_root, "requirements.in")))
    installed = _installed_versions(python)
    return {
        "declared": {d: declared.get(d) for d in _TRACKED_DISTS},
        "installed": installed,
        "zorch_commit": parse_module_pins(
            _read(os.path.join(repo_root, "MODULE.bazel"))
        ).get("zorch"),
        # A venv that does not match the lock file is the state where the
        # number is measured against something the repo never declared.
        "lockstep": all(
            declared.get(d) in (None, installed.get(d)) for d in _TRACKED_DISTS
        ),
    }


def _source(repo_root: str) -> dict[str, Any]:
    return {
        "sha": _run(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root),
        # Untracked files are excluded: the bench writes scratch into the tree
        # and a run is not less trustworthy for it. A modified tracked file is
        # a different story, so `--untracked-files=no` is the right question.
        "dirty": bool(
            _run(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root
            )
        ),
    }


def collect(repo_root: str, python: str) -> dict[str, Any]:
    """The full fingerprint for a measurement taken now, in this environment.

    `python` names the interpreter the bench will run under, which is the only
    one whose installed wheels matter — asking the interpreter collecting the
    fingerprint would describe the wrong venv.
    """
    device = _device()
    return {
        "schema": SCHEMA_VERSION,
        "toolchain": _toolchain(device.get("compute_capability")),
        "device": device,
        "pins": _pins(repo_root, python),
        "overrides": _overrides(repo_root),
        "runtime": {v: os.environ.get(v) for v in _RUNTIME_VARS},
        "source": _source(repo_root),
    }
