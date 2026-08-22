# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Two-sided m32 baseline: this prover against flock's hand-written CUDA prover,
interleaved on one box, with every asymmetry between them named and priced.

The headline "2.3x behind, at 40% of DRAM peak against their 96%" was assembled
by hand from two runs on two different days. That is enough to open an issue and
not enough to steer work off, for the reason `docs/measurement.md` gives at
length: on this box the same binary reads 13-19% apart across processes, so an
un-repeated pair cannot separate a real ratio from process noise. This makes the
pair reproducible in one command.

**What it corrects, and why a raw ratio is wrong without it.** Their
`bench_ligerito` and our `prove_phase_bench` do not measure the same work:

- **Fold PoW.** Their bench runs "grinding OFF": it calls `grind_pow(0)`, the
  unconditional 0-bit query grind, and performs no fold grinds at all. Our m32
  golden carries `grinding_bits [0]*6` — identical to theirs — but
  `fold_grinding_bits [19, 14, 11, 8, 6, 4]`, which is 21 real searches worth
  ~1.07M expected hash attempts, 97% of them in level 0, all inside `open`.
  Charging that to a prover gap compares different work. Corrected by *measuring*
  it: the same prove runs with and without the fold schedule, and the difference
  is the price. (The draft that opened this line assumed "order ~2 ms" for it;
  nobody had measured it.)
- **Witness generation.** Their wall includes it; our phase split starts after
  it. Corrected by subtracting theirs, so neither side counts it.
- **Bench-only input fill.** Their harness times its own input fill and prints
  the figure excluding it. Subtracted.
- **Wall definition.** They synchronize the device at phase boundaries, so their
  wall is comparable to our *barriered* split, not to `--throughput`. This runs
  ours barriered by default for that reason.

**What it deliberately does not do.** It does not claim the corrected pair is
noise-free, and it does not profile. Bytes and achieved bandwidth need ncu (see
`docs/measurement.md` for the `dram__bytes_op_read` naming trap); this reports
walls only, which is what a like-for-like ratio needs.

Run — the toolchain rules in `docs/measurement.md` apply unchanged, and are
NOT set here on purpose, since a harness that quietly exports `CUDA_ROOT` is how
the mixed-toolchain state gets reached:

    export CUDA_ROOT=/path/to/cuda-13.3
    export PATH="$CUDA_ROOT/bin:$PATH"
    export FRX_PLATFORMS=cuda,cpu FRX_ENABLE_X64=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    unset JAX_PLATFORMS JAX_ENABLE_X64 XLA_PYTHON_CLIENT_ALLOCATOR
    PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv>/bin/python \
        python/flock_zorch/testing/rival_compare.py \
        --rival-bin /path/to/flock/cuda-ghash/bench_ligerito

The rival binary's path is a required flag (or `FLOCK_CUDA_BENCH`) with no
default: it lives in a different repo, so any path baked in here would be one
contributor's checkout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Our phase names. Theirs map on as `l0-commit -> commit` plus the two that share
# a name; `open` is the leading figure on their `open ... ms |` line
# (`Phase::open_phase()`), not the `open` bucket further along it, which is the
# multiproof sub-step. `parse_rival` does that mapping where it reads them.
PHASES = ("commit", "zerocheck", "lincheck", "open")

# Marker `prove_phase_bench --json` prefixes its machine-readable line with.
BENCH_JSON_MARK = "##bench-json## "

_RE_CONFIG = re.compile(
    r"log_n=(?P<log_n>\d+)\s+initial_k=(?P<initial_k>\d+)\s+r=(?P<r>\d+)\s+"
    r"k_rec=(?P<k_rec>\d+).*?queries=(?P<queries>[\d,]+)"
)
_RE_OPEN = re.compile(r"^\s*open\s+(?P<open>[\d.]+)\s+ms\s+\|")
_RE_CHAIN = re.compile(
    r"resident chain:\s+witness-gen\s+(?P<witness>[\d.]+)\s+"
    r"l0-commit\s+(?P<l0commit>[\d.]+)\s+zerocheck\s+(?P<zerocheck>[\d.]+)\s+"
    r"lincheck\s+(?P<lincheck>[\d.]+)\s+eq-build\s+(?P<eq_build>[\d.]+)\s+ms"
)
_RE_WALL = re.compile(
    r">>>\s+prove wall\s+(?P<wall>[\d.]+)\s+ms\s+\((?P<excl_fill>[\d.]+)\s+"
    r"excl\. bench fill\)\s+\|\s+phase total\s+(?P<total>[\d.]+)\s+ms\s+\|\s+"
    r"unattributed\s+(?P<unattributed>[-\d.]+)\s+ms\s+\((?P<pct>[+-][\d.]+)%\)"
)


class ParseError(RuntimeError):
    """The rival printed something this cannot read.

    Its own class because the failure mode being guarded is a *silent* one: a
    regex that stops matching after an upstream reformat would otherwise leave
    a field at a default and produce a plausible, wrong ratio.
    """


class Contended(RuntimeError):
    """An arm hit the card while another process held it.

    Separate from a real failure because it is the expected case here, not an
    error: sibling lanes on this box take the card for ~30-60s at a time, so the
    gap between the idle check and the process actually reaching the device is
    wide enough to lose. The arm is retried rather than abandoned, and never
    recorded — a contended sample is not a slow sample, it is an OOM and an
    inflated wall and a huge spread at once (`docs/measurement.md`).
    """


# ------------------------------------------------------------------- the rival


@dataclass(frozen=True)
class RivalRun:
    """One `bench_ligerito <preset>` run, as printed."""

    wall_ms: float
    excl_fill_ms: float
    phase_total_ms: float
    unattributed_ms: float
    unattributed_pct: float
    phases: dict[str, float]
    witness_ms: float
    eq_build_ms: float
    log_n: int
    initial_k: int
    recursive_steps: int
    k_rec: int
    queries: tuple[int, ...]

    @property
    def comparable_ms(self) -> float:
        """Their wall on our terms: no bench-only input fill, no witness-gen.

        Both subtractions are of quantities their own harness reports, so this
        stays a measurement rather than a model.
        """
        return self.excl_fill_ms - self.witness_ms


def parse_rival(text: str) -> RivalRun:
    """Read one `bench_ligerito` run off its stdout.

    Every field is required. A partial parse is treated as a failure rather than
    filled with a default: the number this feeds is a cross-prover ratio, and a
    zero silently standing in for a phase would move it without looking wrong.
    """

    def need(rx: re.Pattern, what: str) -> re.Match:
        m = rx.search(text)
        if m is None:
            raise ParseError(
                f"no {what} line in the rival's output — it likely changed "
                f"format. Pattern: {rx.pattern!r}"
            )
        return m

    cfg = need(_RE_CONFIG, "config")
    chain = need(_RE_CHAIN, "resident-chain")
    wall = need(_RE_WALL, "prove-wall")
    # `open` is line-anchored, so search the lines rather than the blob.
    opens = [m for m in (_RE_OPEN.match(ln) for ln in text.splitlines()) if m]
    if not opens:
        raise ParseError("no `open ... ms |` phase line in the rival's output")

    return RivalRun(
        wall_ms=float(wall["wall"]),
        excl_fill_ms=float(wall["excl_fill"]),
        phase_total_ms=float(wall["total"]),
        unattributed_ms=float(wall["unattributed"]),
        unattributed_pct=float(wall["pct"]),
        phases={
            "commit": float(chain["l0commit"]),
            "zerocheck": float(chain["zerocheck"]),
            "lincheck": float(chain["lincheck"]),
            "open": float(opens[0]["open"]),
        },
        witness_ms=float(chain["witness"]),
        eq_build_ms=float(chain["eq_build"]),
        log_n=int(cfg["log_n"]),
        initial_k=int(cfg["initial_k"]),
        recursive_steps=int(cfg["r"]),
        k_rec=int(cfg["k_rec"]),
        queries=tuple(int(q) for q in cfg["queries"].split(",")),
    )


# --------------------------------------------------------------- instance gate


def instance_mismatches(ours: dict, theirs: RivalRun) -> list[str]:
    """Structural differences between the two configs, as human sentences.

    Names prove nothing — `m32` on one side and `fast32` on the other are just
    labels — so the comparison is gated on the constants that actually define
    the instance. `docs/measurement.md` and the perf-gap method both require
    this before any ratio is quoted.
    """
    ks = list(ours["recursive_ks"])
    checks = [
        ("log_n", ours["log_n"], theirs.log_n),
        ("initial_k", ours["initial_k"], theirs.initial_k),
        ("recursive_steps", len(ks), theirs.recursive_steps),
        ("queries", tuple(ours["queries"]), theirs.queries),
    ]
    out = [f"{n}: ours {a!r} vs theirs {b!r}" for n, a, b in checks if a != b]
    if ks and len(set(ks)) == 1 and ks[0] != theirs.k_rec:
        out.append(f"k_rec: ours {ks[0]!r} vs theirs {theirs.k_rec!r}")
    elif len(set(ks)) > 1:
        out.append(f"k_rec: ours varies {ks!r}, theirs reports one {theirs.k_rec!r}")
    return out


# ------------------------------------------------------------------ card gate


def busy_pids() -> list[int]:
    """PIDs of compute processes on the card, or `[]` when the probe fails.

    A failed probe never blocks: refusing on an unreadable `nvidia-smi` would
    stop a CPU run for no reason. Matches `prove_phase_bench.gpu_provenance`.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except Exception:
        return []
    return [int(ln) for ln in out.split() if ln.strip().isdigit()]


def wait_for_card(timeout_s: float, poll_s: float = 10.0) -> None:
    """Block until no other compute process holds the card.

    This waits rather than refusing outright — unlike the single-shot benches —
    because an interleaved run takes minutes and a sibling lane on this box
    routinely holds ~24 GiB for a stretch. Starting an arm against that does not
    merely add noise: it produces an OOM, an inflated wall and a huge spread at
    once, and all three read as properties of whatever is under test rather
    than of the neighbour (`docs/measurement.md`).
    """
    deadline = time.monotonic() + timeout_s
    announced = False
    while pids := busy_pids():
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"REFUSING to measure: card still held by {pids} after "
                f"{timeout_s:.0f}s. Raise --wait-card, or wait for the lane."
            )
        if not announced:
            print(f"  card held by {pids}; waiting...", flush=True)
            announced = True
        time.sleep(poll_s)


# ---------------------------------------------------------------------- arms


# Every way either side reports running out of device memory. Matched as a set
# because the two provers surface it through different layers — frx's allocator,
# XLA's autotuner, and the rival's raw driver call each word it differently.
_OOM_MARKERS = (
    "CUDA_ERROR_OUT_OF_MEMORY",
    "RESOURCE_EXHAUSTED",
    "out of memory",
)


def _oom_signature(text: str) -> str | None:
    """The first out-of-memory marker present in `text`, or None."""
    low = text.lower()
    return next((m for m in _OOM_MARKERS if m.lower() in low), None)


def run_arm(arm: "Arm", args):
    """Gate on an idle card, run one arm, and retry if it lost the race anyway.

    The gate and the arm's first device touch are seconds apart — process start,
    frx import, golden ingest — which is ample for a sibling lane to take the
    card in between. Retrying is not papering over a flaky measurement: a
    contended arm is discarded, never recorded, so the samples that survive are
    all uncontended ones.
    """
    for attempt in range(1, args.retries + 1):
        wait_for_card(args.wait_card)
        try:
            out = arm.run()
        except Contended as e:
            print(f"  contended ({e}); retry {attempt}/{args.retries}", flush=True)
            continue
        # Re-check AFTER the arm. A neighbour that starts mid-arm inflates the
        # wall without ever OOMing, so neither the pre-gate nor the OOM check
        # sees it — one such sample read 81.40 ms against a 55-63 ms norm.
        # min-of-N would demote it, but only if some other sample is clean;
        # discarding it outright is what makes "every recorded sample ran on an
        # idle card" true rather than likely. It still cannot see a neighbour
        # that both starts and exits inside the arm.
        if late := busy_pids():
            print(
                f"  card taken during the arm by {late}; discarding "
                f"({attempt}/{args.retries})",
                flush=True,
            )
            continue
        return out
    raise SystemExit(
        f"REFUSING to measure: {arm.label} lost the card on every one of "
        f"{args.retries} attempts. The box is too busy to measure on right now."
    )


def wall_of(out) -> float:
    """The wall in ms of either side's result object — one accessor so the
    interleave loop and the min-picker cannot disagree about where it lives."""
    return out.wall_ms if isinstance(out, RivalRun) else out["wall_ms"]


@dataclass
class Arm:
    """One thing being timed, and the samples collected for it."""

    key: str
    label: str
    run: Callable[[], object]
    samples: list[float] = field(default_factory=list)

    def best(self) -> float:
        return min(self.samples)

    def spread_pct(self) -> float:
        """Max-to-min spread as a percentage of min. Reported next to every
        figure because a min without one invites reading noise as a delta."""
        return 100.0 * (max(self.samples) - min(self.samples)) / min(self.samples)


def run_ours(args, *, no_fold_grind: bool) -> dict:
    """One `prove_phase_bench` process, returning its parsed JSON payload.

    A fresh process per sample, not a loop inside one, for two reasons. It is
    the only way the rival can run in between — a resident frx process holds
    device memory that `bench_ligerito`'s ~25 GiB allocation then cannot get —
    and cross-process variation is the dominant error term on this box, so
    sampling within one process would report a spread that hides it.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "prove_phase_bench.py"),
        args.circuit,
        "--json",
        "--runs",
        str(args.runs),
        "--hash",
        args.hash,
    ]
    if args.golden:
        cmd += ["--golden", args.golden]
    if no_fold_grind:
        cmd += ["--no-fold-grind"]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    if "other compute process" in proc.stderr:
        raise Contended("prove_phase_bench's own guard refused")
    if oom := _oom_signature(proc.stdout + proc.stderr):
        # The guard runs before frx allocates, so an arm can clear it and still
        # lose the card seconds later, during autotuning or the first commit.
        # `docs/measurement.md`: an OOM during an A/B is contention until the
        # card is proven idle. This prove is known to fit an idle 5090.
        raise Contended(f"our arm OOM'd ({oom})")
    line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith(BENCH_JSON_MARK)), None
    )
    if line is None:
        raise ParseError(
            "prove_phase_bench emitted no JSON line "
            f"(exit {proc.returncode}).\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return json.loads(line[len(BENCH_JSON_MARK) :])


def run_rival(args) -> RivalRun:
    """One `bench_ligerito <preset>` process.

    Run from the binary's own directory: it loads `blake3_lincheck_matrices.bin`
    by relative path and dies if the cwd is elsewhere.
    """
    binary = Path(args.rival_bin).resolve()
    proc = subprocess.run(
        [str(binary), args.rival_preset],
        capture_output=True,
        text=True,
        cwd=binary.parent,
        timeout=args.timeout,
    )
    if oom := _oom_signature(proc.stdout + proc.stderr):
        # Checked before the exit code: an OOM here exits non-zero, and on this
        # box an OOM during an A/B is contention until the card is proven idle.
        raise Contended(f"rival OOM'd ({oom})")
    if proc.returncode != 0:
        raise SystemExit(
            f"rival exited {proc.returncode}.\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return parse_rival(proc.stdout)


# --------------------------------------------------------------------- report


def _ratio(a: float, b: float) -> str:
    return f"{b / a:.2f}x" if a else "   n/a"


def report(ours_grind, ours_plain, rival, args) -> None:
    """Print the comparison, asymmetries first."""
    theirs, mine_grind, mine_plain = rival[0], ours_grind[0], ours_plain[0]

    print("\n" + "=" * 72)
    print(f"m{mine_grind['m']} | {args.circuit} | hash arm {mine_grind['hash_arm']}")
    print("=" * 72)

    mismatches = instance_mismatches(mine_grind, theirs)
    if mismatches:
        raise SystemExit(
            "REFUSING to report a ratio: the two sides are not the same "
            "instance.\n  " + "\n  ".join(mismatches)
        )
    print(
        f"instance verified structurally: log_n={theirs.log_n} "
        f"initial_k={theirs.initial_k} r={theirs.recursive_steps} "
        f"k_rec={theirs.k_rec} queries={','.join(map(str, theirs.queries))}"
    )
    if abs(theirs.unattributed_pct) > 1.0:
        print(
            f"  WARNING rival reports {theirs.unattributed_ms:+.2f} ms "
            f"({theirs.unattributed_pct:+.2f}%) unattributed — its own phase "
            "split does not account for its wall, so the per-phase rows below "
            "inherit that gap."
        )

    # Both of our arms are shown per phase, not just the shipping one. Their
    # bench performs no fold PoW, so an `open` row comparing their open against
    # our as-shipped open charges them for work they never did — which is the
    # confusion this whole harness exists to remove. The ratio column is taken
    # against the fold-PoW-off arm for that reason, and the `delta` column says
    # where the dropped PoW actually landed.
    print("\n-- phases (ms, best of each side's samples) " + "-" * 28)
    print(
        f"  {'phase':<12}{'flock':>9}{'ours':>9}{'ours-noPoW':>12}"
        f"{'PoW':>8}{'ratio':>8}"
    )
    for p in PHASES:
        t = theirs.phases[p]
        shipped, plain = mine_grind["phases"][p], mine_plain["phases"][p]
        print(
            f"  {p:<12}{t:>9.2f}{shipped:>9.2f}{plain:>12.2f}"
            f"{shipped - plain:>+8.2f}{_ratio(t, plain):>8}"
        )
    print("  (ratio is theirs vs our fold-PoW-off arm — the like-for-like pair)")

    grind = mine_grind["wall_ms"] - mine_plain["wall_ms"]
    dropped = mine_plain["fold_pow_dropped"]
    print("\n-- asymmetries " + "-" * 57)
    print(
        f"  fold PoW      ours does {dropped['grinds']} grinds "
        f"({dropped['expected_attempts']:,} expected attempts = "
        f"{dropped['windowed_hashes']:,} hashes evaluated), theirs 0 "
        f"-> MEASURED {grind:+.2f} ms"
    )
    print(
        f"  witness-gen   theirs includes {theirs.witness_ms:.2f} ms, "
        "our split excludes it -> subtracted from theirs"
    )
    print(
        f"  bench fill    theirs includes "
        f"{theirs.wall_ms - theirs.excl_fill_ms:.2f} ms of harness input fill "
        "-> subtracted from theirs"
    )
    print(
        "  wall def      theirs syncs per phase -> compared against our "
        "barriered split, not --throughput"
    )

    print("\n-- like-for-like " + "-" * 55)
    rows = [
        ("flock, as reported", theirs.wall_ms, None),
        ("flock, comparable", theirs.comparable_ms, None),
        ("ours, as shipped", mine_grind["wall_ms"], ours_grind[1]),
        ("ours, fold PoW off", mine_plain["wall_ms"], ours_plain[1]),
    ]
    for name, ms, spread in rows:
        hs = mine_grind["hashes"] * 1e3 / ms
        tail = f"  (spread {spread:.1f}%)" if spread is not None else ""
        print(f"  {name:<22}{ms:>9.2f} ms{hs / 1e6:>9.2f}M hash/s{tail}")

    lfl = mine_plain["wall_ms"] / theirs.comparable_ms
    print(
        f"\n  LIKE-FOR-LIKE RATIO  {lfl:.2f}x  "
        "(ours fold-PoW-off / theirs comparable)"
    )
    raw = mine_grind["wall_ms"] / theirs.wall_ms
    print(f"  raw ratio            {raw:.2f}x  (both as each harness reports)")
    print(
        f"\n  samples: {args.rounds} interleaved round(s), best-of-{args.runs} "
        "within each process. Every arm gated on an idle card."
    )


# ----------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rival-bin",
        default=os.environ.get("FLOCK_CUDA_BENCH"),
        help="path to flock's `cuda-ghash/bench_ligerito` (or FLOCK_CUDA_BENCH). "
        "Required, and deliberately without a default — it lives in another "
        "repo, so any path here would be one checkout's",
    )
    ap.add_argument("--rival-preset", default="fast32", help="the rival's preset")
    ap.add_argument("--circuit", default="blake3")
    ap.add_argument(
        "--golden",
        default="blake3_ligerito_golden_m32.bin",
        help="golden under artifacts/",
    )
    ap.add_argument("--hash", default="sha256", choices=("sha256", "blake3"))
    ap.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="interleaved rounds. Each round runs every arm, and the arm order "
        "reverses on alternate rounds, so no arm keeps a favourable slot as the "
        "card warms",
    )
    ap.add_argument(
        "--runs", type=int, default=3, help="best-of, within each of our processes"
    )
    ap.add_argument(
        "--wait-card",
        type=float,
        default=900.0,
        help="seconds to wait for an idle card before each arm",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=5,
        help="attempts per arm before giving up, when the card is lost to a "
        "sibling lane between the idle gate and the arm's first allocation",
    )
    ap.add_argument(
        "--timeout", type=float, default=1800.0, help="per-subprocess timeout, seconds"
    )
    args = ap.parse_args()

    if not args.rival_bin:
        ap.error("--rival-bin (or FLOCK_CUDA_BENCH) is required")
    if not Path(args.rival_bin).is_file():
        ap.error(f"--rival-bin is not a file: {args.rival_bin}")
    if args.rounds < 1:
        ap.error("--rounds must be >= 1")

    arms = [
        Arm(
            "ours_grind",
            "ours (as shipped)",
            lambda: run_ours(args, no_fold_grind=False),
        ),
        Arm(
            "ours_plain",
            "ours (fold PoW off)",
            lambda: run_ours(args, no_fold_grind=True),
        ),
        Arm("rival", "flock cuda-ghash", lambda: run_rival(args)),
    ]
    results: dict[str, list] = {a.key: [] for a in arms}
    gated = False

    for rnd in range(args.rounds):
        order = arms if rnd % 2 == 0 else list(reversed(arms))
        for arm in order:
            print(f"[round {rnd + 1}/{args.rounds}] {arm.label}", flush=True)
            out = run_arm(arm, args)
            ms = wall_of(out)
            arm.samples.append(ms)
            results[arm.key].append(out)
            print(f"  {ms:.2f} ms", flush=True)

            # Gate the instance as soon as one sample of each side exists,
            # rather than at report time: a mismatch makes every later arm
            # worthless, and on a contended box those arms cost many minutes
            # each of waiting for the card.
            if not gated and results["rival"] and results["ours_grind"]:
                gated = True
                if bad := instance_mismatches(
                    results["ours_grind"][0], results["rival"][0]
                ):
                    raise SystemExit(
                        "REFUSING to measure further: the two sides are not "
                        "the same instance.\n  " + "\n  ".join(bad)
                    )

    picked = {
        arm.key: (
            min(results[arm.key], key=wall_of),
            arm.spread_pct() if len(arm.samples) > 1 else None,
        )
        for arm in arms
    }

    report(picked["ours_grind"], picked["ours_plain"], picked["rival"], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
