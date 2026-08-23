# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Publishing a measurement as a check run — the part no benchmark owns.

A check run on the measured commit is where a number can be found again:
`GET /commits/{sha}/check-runs` replays a whole trajectory with no store to
keep in sync. None of that reasoning is specific to this prover, so it lives
here and `scripts/bench_ledger.py` supplies only the flock-specific half —
how to run the bench and what one measurement of it looks like.

**Adding a second consumer.** Everything below is repo-agnostic. A new repo
supplies four things and reuses the rest:

1. a way to run its benchmark, returning an `Outcome`;
2. a record in the shape `{"benchmarks": [{...}], "window": {...}}`, where the
   benchmark entry carries `name`, `variant`, `metrics` (throughput in units
   per second, latency in ms — the convention `artifacts/job-*.json` set),
   a free-form `instance` describing what was measured, and `env` (a
   `_fingerprint.collect()` result);
3. a `subject` for the headline (this repo passes `m32`);
4. the minimum toolchain its numbers are meaningless below, if any — that is
   the one genuinely per-repo part of the fingerprint (here, the CUDA 13.3
   that assembles clmad).

When that second consumer arrives, this module and `_fingerprint.py` move to
a reusable workflow in `fractalyze/.github` unchanged; the seam is already
where the split would go.

**The wire contract is `LEDGER_MARK`.** A consumer (today the work-map view's
`fetch_measurements.py`) finds that marker in `output.summary` and parses the
fenced JSON after it. Changing the record's meaning means bumping the marker,
or readers silently mix two schemas — the exact failure the fingerprint exists
to prevent.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, NamedTuple

from flock_zorch.testing._fingerprint import blocking, drift

# Opens the structured block inside `output.summary`; everything else in the
# summary is for a human reading the Checks tab. Bump the version when the
# record's meaning changes.
LEDGER_MARK = "<!-- bench-ledger:v1 -->"

# Exit code meaning "measured nothing, on purpose". Matches the convention
# `prove_phase_bench.py` already uses for its pre-flight refusals.
REFUSED = 2


class Outcome(NamedTuple):
    """What one attempt to measure produced. Exactly one field is set.

    The three cases exist so that "we could not measure" never reads as "the
    code regressed": only `error` is a failure, and a refusal still leaves the
    build green because GPU contention on a shared box must not turn a branch
    red.
    """

    record: dict[str, Any] | None = None
    refusal: str | None = None
    error: str | None = None


def load_baseline(path: str | None) -> dict[str, Any] | None:
    """The environment of a previous record, for the comparability verdict."""
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        previous = json.load(f)
    return {**previous["benchmarks"][0]["env"], "window": previous.get("window")}


def comparability(
    baseline: dict[str, Any] | None,
    env: dict[str, Any],
    window: dict[str, Any] | None = None,
) -> list[str]:
    """Why this may not be compared against the baseline — empty if it may.

    The question a trajectory actually raises is not "did the number move" but
    "am I allowed to subtract these two", and that is the one nobody answers
    from memory correctly.

    `window` is optional because a caller deciding whether to spend GPU time at
    all only knows the environment yet. Comparing without it drops the
    baseline's window rather than diffing against a missing key, which would
    otherwise report every pre-flight check as drifted.
    """
    if baseline is None:
        return []
    if window is None:
        baseline = {k: v for k, v in baseline.items() if k != "window"}
        current = dict(env)
    else:
        current = {**env, "window": window}
    return [d.describe() for d in blocking(drift(baseline, current))]


def _si(value: float) -> str:
    """A throughput a human reads at a glance, at whatever scale it lands on.

    Fixed millions would render a CPU arm's 701/s as `0.00M`, which reads as a
    broken run rather than a slow one.
    """
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= scale:
            return f"{value / scale:.2f}{suffix}"
    return f"{value:.0f}"


def _short(sha: str | None) -> str:
    return sha[:8] if sha else "unknown"


def summarise(
    record: dict[str, Any], incomparable: list[str], subject: str
) -> tuple[str, str]:
    """`(title, summary)` for a measurement — a headline, then the block.

    The table is built entirely from the fingerprint, so it needs to know
    nothing about the benchmark: it answers "may I compare this against the
    last one?", which is the question a reader has and the one that costs a
    session when it is answered from memory.
    """
    bench = record["benchmarks"][0]
    env = bench["env"]
    tool, dev, pins = env["toolchain"], env["device"], env["pins"]
    metrics, window = bench["metrics"], record["window"]

    title = (
        f"{_si(metrics['throughput'])} /s — {metrics['latency']:.2f} ms at {subject}"
    )
    rows = [
        ("ptxas / nvlink", f"{tool['ptxas']} / {tool['nvlink']}"),
        (
            "card",
            f"{dev['name']} · driver {dev['driver']}"
            + (f" · gpu {dev['pinned_to']}" if dev.get("pinned_to") else ""),
        ),
        ("frx", pins["installed"].get("frx") or "unknown"),
        (
            "overrides",
            ", ".join(
                f"{m} @ {_short(o['head'])}"
                f"{'' if o['matches_pin'] else ' (NOT the declared pin)'}"
                for m, o in env["overrides"].items()
            )
            or "none",
        ),
        ("allocator", env["runtime"]["XLA_PYTHON_CLIENT_ALLOCATOR"] or "default"),
        ("window", f"{window['mode']}, best-of-{window['runs']}"),
        ("lockstep", "yes" if pins["lockstep"] else "NO — venv differs from the lock"),
    ]
    # Only for repos that pin a source dep this way; absent elsewhere rather
    # than rendered as "None".
    if pins.get("zorch_commit"):
        rows.insert(2, ("zorch pin", _short(pins["zorch_commit"])))

    verdict = (
        "comparable against the previous record"
        if not incomparable
        else "**NOT comparable** against the previous record: "
        + "; ".join(incomparable)
    )
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return title, (
        f"**{title}** · {bench['name']} · {window['mode']}, "
        f"best-of-{window['runs']}\n\n"
        f"{verdict}\n\n"
        f"| | |\n|---|---|\n{table}\n\n"
        f"{LEDGER_MARK}\n```json\n{json.dumps(record, indent=2)}\n```\n"
    )


def summarise_outcome(outcome: Outcome, detail: str = "") -> tuple[str, str]:
    """`(title, summary)` for a run that produced no measurement."""
    refused = outcome.refusal is not None
    title = "not measured" if refused else "the bench failed"
    body = (
        "A contended card or a capped toolchain means this commit was not "
        "measured, not that it got slower. No point is added to the "
        "trajectory.\n"
        if refused
        else ""
    )
    reason = outcome.refusal if refused else (outcome.error or "no reason given")
    tail = f"\n```\n{detail.strip()[-2000:]}\n```\n" if detail.strip() else ""
    return title, f"**{title}** — {reason}\n\n{body}{tail}"


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


def publish(
    outcome: Outcome,
    *,
    name: str,
    head_sha: str,
    subject: str,
    baseline: dict[str, Any] | None = None,
    detail: str = "",
    out: str | None = None,
    check_run: bool = False,
) -> int:
    """Record one outcome and return the process exit code.

    Writes the record to `--out` only when there is one, so a refusal cannot
    overwrite what the next run is compared against.
    """
    if outcome.record is not None:
        incomparable = comparability(
            baseline,
            outcome.record["benchmarks"][0]["env"],
            outcome.record["window"],
        )
        conclusion = "success"
        title, summary = summarise(outcome.record, incomparable, subject)
        rendered = json.dumps(outcome.record, indent=2)
        if out:
            with open(out, "w", encoding="utf-8") as f:
                f.write(rendered + "\n")
        print(rendered)
    else:
        conclusion = "neutral" if outcome.refusal is not None else "failure"
        title, summary = summarise_outcome(outcome, detail)
        print(summary, file=sys.stderr)

    if check_run:
        token = os.environ.get("GITHUB_TOKEN")
        repo = os.environ.get("GITHUB_REPOSITORY")
        if not token or not repo:
            print(
                "--check-run needs GITHUB_TOKEN and GITHUB_REPOSITORY (it is "
                "meant to run inside Actions, whose token is a GitHub App — a "
                "PAT gets 403, check runs are App-only)",
                file=sys.stderr,
            )
            return 1
        try:
            url = post_check_run(
                repo, token, name, head_sha, conclusion, title, summary
            )
        except urllib.error.HTTPError as e:
            print(
                f"check-run POST failed: {e.code} "
                f"{e.read().decode(errors='replace')[:500]}",
                file=sys.stderr,
            )
            return 1
        print(f"check run: {url}", file=sys.stderr)

    # A refusal is not a failing build: the whole point of `neutral` is that a
    # busy GPU does not turn a branch red.
    return 0 if conclusion in ("success", "neutral") else 1
