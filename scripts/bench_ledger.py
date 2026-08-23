#!/usr/bin/env python3
# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Measure this prover once and record it where the trajectory survives.

A number in a session log is a number nobody can compare later: the box, the
toolchain and the pins that produced it are gone by the time anyone asks. This
runs `prove_phase_bench.py`, attaches the full environment fingerprint, and
hands the result to `_ledger.publish`, which writes it as a check run on the
measured commit.

This file is only the flock-specific half — how to run the bench and what one
measurement of it looks like. Everything about publishing (the check run, the
comparability verdict, the conclusion semantics, the wire contract) lives in
`flock_zorch.testing._ledger`, which knows nothing about this prover.

Usage (from the repo root, with the toolchain preamble README describes):

  scripts/bench_ledger.py blake3 --golden blake3_ligerito_golden_m32.bin \\
      --runs 10 --out record.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from flock_zorch.testing._fingerprint import collect, publish_refusal  # noqa: E402
from flock_zorch.testing._ledger import (  # noqa: E402
    REFUSED,
    Outcome,
    comparability,
    load_baseline,
    publish,
)

# `prove_phase_bench.py` prints its machine-readable line under this prefix,
# alongside the human table and the toolchain notices.
JSON_MARK = "##bench-json## "

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_bench(
    python: str, circuit: str, golden: str | None, runs: int
) -> tuple[int, str, str]:
    """`(exit code, stdout, stderr)` of one `prove_phase_bench --throughput` run.

    `--throughput` and not the phase split: the barriered mode serialises work
    the throughput path overlaps and reads ~14% slower, so it is an attribution
    tool and never the number a goal is stated in (`docs/measurement.md`).
    """
    cmd = [
        python,
        os.path.join("python", "flock_zorch", "testing", "prove_phase_bench.py"),
        circuit,
        "--throughput",
        "--json",
        "--runs",
        str(runs),
    ]
    if golden:
        cmd += ["--golden", golden]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    return proc.returncode, proc.stdout, proc.stderr


def parse_bench_json(stdout: str) -> dict[str, Any] | None:
    """The `##bench-json##` payload, or None if the run printed no measurement."""
    for line in stdout.splitlines():
        if line.startswith(JSON_MARK):
            return json.loads(line[len(JSON_MARK) :])
    return None


def to_record(
    payload: dict[str, Any], fingerprint: dict[str, Any], runs: int
) -> dict[str, Any]:
    """One measurement in the shape the existing `artifacts/job-*.json` use.

    The envelope (`benchmarks[]` with `suite`/`name`/`variant`/`metrics`) is
    kept as it was so the unit conventions carry over — throughput in hash/s,
    latency in ms. What changes is `env`: it used to be three fields (ptxas,
    driver, allocator), which is not enough to decide whether two rows may be
    compared, so it is now the whole fingerprint and the window that produced
    the number.
    """
    wall_ms = payload["wall_ms"]
    hashes = payload["hashes"]
    return {
        "benchmarks": [
            {
                "suite": "prove_phase_bench",
                "name": f"{payload['circuit']}_m{payload['m']}",
                "variant": payload["mode"],
                "metrics": {
                    "throughput": hashes / (wall_ms / 1e3),
                    "latency": wall_ms,
                },
                # The instance, carried next to the timing because a wall is
                # only meaningful once both sides are shown to be the same
                # circuit — a consumer that infers it from a name will
                # eventually infer it wrong.
                "instance": {
                    "m": payload["m"],
                    "log_n": payload["log_n"],
                    "hashes": hashes,
                    "initial_k": payload["initial_k"],
                    "recursive_ks": payload["recursive_ks"],
                    "queries": payload["queries"],
                    "hash_arm": payload["hash_arm"],
                },
                "env": fingerprint,
            }
        ],
        # Hoisted out of `env` because it is the first thing a comparison has
        # to agree on and the last thing anyone remembers to write down.
        "window": {
            "mode": payload["mode"],
            "runs": runs,
            "processes": 1,
            "fold_pow_dropped": payload["fold_pow_dropped"],
        },
    }


def _refusal_reason(stderr: str) -> str:
    """The bench's own refusal line, which already names what to do about it."""
    for line in stderr.splitlines():
        if line.startswith("REFUSING to measure:"):
            return line[len("REFUSING to measure:") :].strip()
    return "the bench refused to measure and gave no reason"


def measure(
    args: argparse.Namespace, fingerprint: dict[str, Any]
) -> tuple[Outcome, str]:
    """`(outcome, stderr)` for one attempt at this prover's benchmark."""
    # Routed through the same refusal path as the bench's own guards, so a
    # toolchain that cannot be trusted reads as "not measured" rather than as
    # a slow commit — and costs no GPU time finding that out.
    gate = publish_refusal(fingerprint) if args.require_toolchain else None
    if gate:
        return Outcome(refusal=gate), ""

    code, stdout, stderr = run_bench(args.python, args.circuit, args.golden, args.runs)
    payload = parse_bench_json(stdout)
    if code == REFUSED:
        return Outcome(refusal=_refusal_reason(stderr)), stderr
    if payload is None:
        detail = stderr or "the bench exited 0 but printed no measurement"
        return Outcome(error=detail.strip()[-500:]), stderr
    return Outcome(record=to_record(payload, fingerprint, args.runs)), stderr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("circuit", nargs="?", default="blake3")
    ap.add_argument("--golden", help="golden filename under artifacts/")
    ap.add_argument("--runs", type=int, default=10, help="best-of-N within one process")
    ap.add_argument("--python", default=".venv/bin/python")
    ap.add_argument("--out", help="write the record here")
    ap.add_argument(
        "--baseline",
        help="a previous record to diff the fingerprint against; the check run "
        "reports whether the two numbers may be compared",
    )
    ap.add_argument(
        "--refuse-on-drift",
        action="store_true",
        help="exit before measuring when the fingerprint has drifted from "
        "--baseline in a way that makes the comparison meaningless. For "
        "interactive use, where the point is not to spend the GPU time; CI "
        "wants the point recorded even when it cannot be compared.",
    )
    ap.add_argument(
        "--check-run",
        action="store_true",
        help="create a GitHub check run (needs Actions' GITHUB_TOKEN — a PAT "
        "gets 403, check runs are App-only)",
    )
    ap.add_argument("--name", help="check-run name; groups the trajectory")
    ap.add_argument("--head-sha", help="commit to attach to (default: the checkout's)")
    ap.add_argument(
        "--require-toolchain",
        action="store_true",
        help="refuse to measure unless the toolchain is provably able to "
        "assemble clmad. The bench's own guard fails open on an unreadable "
        "probe so it never blocks a human; an unattended run has no reader, "
        "and a capped toolchain publishes a ~15x-slow number as a success.",
    )
    args = ap.parse_args()

    fingerprint = collect(REPO_ROOT, args.python)
    baseline = load_baseline(args.baseline)
    subject = (
        re.sub(r"^blake3_ligerito_golden_?|\.bin$", "", args.golden or "") or "default"
    )
    name = args.name or f"bench ({args.circuit} {subject})"
    sha = args.head_sha or fingerprint["source"]["sha"]

    if args.refuse_on_drift:
        # Priced before the GPU time rather than after: the point of asking
        # interactively is not to spend minutes producing a number that cannot
        # be compared to anything. Only the environment is known this early —
        # window drift still shows up in the check run afterwards.
        if drifted := comparability(baseline, fingerprint):
            print(
                "REFUSING to measure: the environment has drifted from the "
                "baseline, so the comparison would say nothing:\n  "
                + "\n  ".join(drifted),
                file=sys.stderr,
            )
            return REFUSED

    outcome, stderr = measure(args, fingerprint)
    return publish(
        outcome,
        name=name,
        head_sha=sha,
        subject=subject,
        baseline=baseline,
        detail=stderr,
        out=args.out,
        check_run=args.check_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
