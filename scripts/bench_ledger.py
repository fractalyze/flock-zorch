#!/usr/bin/env python3
# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Run one benchmark and record it where the trajectory can be reconstructed.

A number in a session log is a number nobody can compare later: the box, the
toolchain and the pins that produced it are gone by the time anyone asks. This
runs `prove_phase_bench.py` and writes the result together with the full
environment fingerprint (`_fingerprint.py`) as a GitHub **check run** on the
measured commit, so `GET /commits/{sha}/check-runs` replays the whole history
with no store of our own to keep in sync.

Check runs can only be created by a GitHub App, so `--check-run` works from
Actions (whose `GITHUB_TOKEN` is one) and not from a personal access token,
which gets a 403. Without the flag the record goes to stdout and `--out`,
which is the useful mode on a workstation.

Conclusions are chosen so that "we could not measure" never reads as "the code
regressed":

  success  a number was produced
  neutral  the bench refused to measure — a contended card or a toolchain that
           cannot assemble clmad. On a shared box this is the common case and
           it is not a failure; GPU contention must not turn a branch red.
  failure  the bench itself broke

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
import urllib.error
import urllib.request
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from flock_zorch.testing._fingerprint import (  # noqa: E402
    blocking,
    collect,
    drift,
    publish_refusal,
)

# `prove_phase_bench.py` prints its machine-readable line under this prefix,
# alongside the human table and the toolchain notices.
JSON_MARK = "##bench-json## "
# Opens the structured block inside `output.summary`. A consumer looks for
# this and reads the fenced JSON that follows; everything else in the summary
# is for a human reading the Checks tab.
LEDGER_MARK = "<!-- bench-ledger:v1 -->"
# `prove_phase_bench` exits 2 for both of its pre-flight refusals.
REFUSED = 2

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


def _si(hash_per_s: float) -> str:
    return f"{hash_per_s / 1e6:.2f}M hash/s"


def _short(sha: str | None) -> str:
    return sha[:8] if sha else "unknown"


def summarise(record: dict[str, Any], comparability: list[str]) -> tuple[str, str]:
    """`(title, summary)` for the check run — a human headline, then the block.

    The table answers "may I compare this against the last one?", which is the
    question a reader actually has and the one that costs a session when it is
    answered from memory.
    """
    bench = record["benchmarks"][0]
    env = bench["env"]
    tool, dev, pins = env["toolchain"], env["device"], env["pins"]
    overrides = env["overrides"]
    metrics, window = bench["metrics"], record["window"]

    title = (
        f"{_si(metrics['throughput'])} — {metrics['latency']:.2f} ms "
        f"at m{bench['instance']['m']}"
    )
    rows = [
        ("ptxas / nvlink", f"{tool['ptxas']} / {tool['nvlink']}"),
        ("card", f"{dev['name']} · driver {dev['driver']}"),
        ("zorch pin", _short(pins["zorch_commit"])),
        ("frx", pins["installed"].get("frx") or "unknown"),
        (
            "overrides",
            ", ".join(
                f"{m} @ {_short(o['head'])}"
                f"{'' if o['matches_pin'] else ' (NOT the declared pin)'}"
                for m, o in overrides.items()
            )
            or "none",
        ),
        ("allocator", env["runtime"]["XLA_PYTHON_CLIENT_ALLOCATOR"] or "default"),
        ("window", f"{window['mode']}, best-of-{window['runs']}"),
        ("lockstep", "yes" if pins["lockstep"] else "NO — venv differs from the lock"),
    ]
    verdict = (
        "comparable against the previous record"
        if not comparability
        else "**NOT comparable** against the previous record: "
        + "; ".join(comparability)
    )
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return title, (
        f"**{title}** · {bench['name']} · {window['mode']}, "
        f"best-of-{window['runs']}\n\n"
        f"{verdict}\n\n"
        f"| | |\n|---|---|\n{table}\n\n"
        f"{LEDGER_MARK}\n```json\n{json.dumps(record, indent=2)}\n```\n"
    )


def post_check_run(
    repo: str,
    token: str,
    name: str,
    sha: str,
    conclusion: str,
    title: str,
    summary: str,
) -> str:
    """Create the check run and return its html_url."""
    body = json.dumps(
        {
            "name": name,
            "head_sha": sha,
            "status": "completed",
            "conclusion": conclusion,
            "output": {"title": title, "summary": summary},
        }
    ).encode()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/check-runs",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return str(json.load(resp)["html_url"])


def _refusal_reason(stderr: str) -> str:
    """The bench's own refusal line, which already names what to do about it."""
    for line in stderr.splitlines():
        if line.startswith("REFUSING to measure:"):
            return line[len("REFUSING to measure:") :].strip()
    return "the bench refused to measure and gave no reason"


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
    baseline = None
    comparability: list[str] = []
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline, encoding="utf-8") as f:
            previous = json.load(f)
        baseline = {
            **previous["benchmarks"][0]["env"],
            "window": previous.get("window"),
        }

    size = (
        re.sub(r"^blake3_ligerito_golden_?|\.bin$", "", args.golden or "") or "default"
    )
    name = args.name or f"bench ({args.circuit} {size})"
    sha = args.head_sha or fingerprint["source"]["sha"]

    # Routed through the same refusal path as the bench's own guards, so a
    # toolchain that cannot be trusted reads as "not measured" rather than as
    # a slow commit — and costs no GPU time finding that out.
    gate = publish_refusal(fingerprint) if args.require_toolchain else None
    if gate:
        code, stdout, stderr = REFUSED, "", f"REFUSING to measure: {gate}\n"
    else:
        code, stdout, stderr = run_bench(
            args.python, args.circuit, args.golden, args.runs
        )
    payload = parse_bench_json(stdout)

    if code == REFUSED or (code != 0 and payload is None):
        refused = code == REFUSED
        reason = _refusal_reason(stderr) if refused else stderr.strip()[-2000:]
        conclusion = "neutral" if refused else "failure"
        title = "not measured" if refused else "the bench failed"
        summary = (
            f"**{title}** — {reason}\n\n"
            + (
                "A contended card or a capped toolchain means this commit was "
                "not measured, not that it got slower. No point is added to "
                "the trajectory.\n"
                if refused
                else ""
            )
            + f"\n```\n{stderr.strip()[-2000:]}\n```\n"
        )
        print(summary, file=sys.stderr)
    else:
        if payload is None:
            print("bench exited 0 but printed no measurement", file=sys.stderr)
            return 1
        record = to_record(payload, fingerprint, args.runs)
        if baseline is not None:
            current = {**fingerprint, "window": record["window"]}
            comparability = [d.describe() for d in blocking(drift(baseline, current))]
            if comparability and args.refuse_on_drift:
                print(
                    "REFUSING to measure: the environment has drifted from the "
                    "baseline, so the comparison would say nothing:\n  "
                    + "\n  ".join(comparability),
                    file=sys.stderr,
                )
                return REFUSED
        conclusion = "success"
        title, summary = summarise(record, comparability)
        out = json.dumps(record, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(out + "\n")
        print(out)

    if args.check_run:
        token = os.environ.get("GITHUB_TOKEN")
        repo = os.environ.get("GITHUB_REPOSITORY")
        if not token or not repo:
            print(
                "--check-run needs GITHUB_TOKEN and GITHUB_REPOSITORY (it is "
                "meant to run inside Actions)",
                file=sys.stderr,
            )
            return 1
        try:
            url = post_check_run(repo, token, name, sha, conclusion, title, summary)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            print(f"check-run POST failed: {e.code} {detail}", file=sys.stderr)
            return 1
        print(f"check run: {url}", file=sys.stderr)

    # A refusal is not a failing build: the whole point of `neutral` is that a
    # busy GPU does not turn a branch red.
    return 0 if conclusion in ("success", "neutral") else 1


if __name__ == "__main__":
    raise SystemExit(main())
