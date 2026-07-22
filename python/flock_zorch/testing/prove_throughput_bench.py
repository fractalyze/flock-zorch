# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sustained hashes/second from N co-scheduled prover processes on one card.

`prove_phase_bench` answers "where does one prove spend its time"; this answers
the throughput question the 5M-BLAKE3/s milestone is stated in: how many hashes
per second the card sustains when proofs are produced back-to-back. The two
differ because a single prove stream leaves the GPU idle roughly half its wall
— the prove is a host-dispatch chain of thousands of small launches, and the
card waits out every inter-launch gap. One process cannot fill those gaps from
inside (the dispatch chain is serial Python; threads serialize on the GIL), but
N *processes* proving independent instances can — their kernels interleave into
each other's gaps.

That only works under **CUDA MPS**. Without it, processes hold separate CUDA
contexts and the driver time-slices them at a granularity far coarser than the
µs-scale gaps being filled — measured on this workload it is a net LOSS (3
plain processes aggregate ~0.9x of one). Under MPS all processes share one
context and kernels co-schedule. Measured on an RTX 5090 at blake3 m=28
(zorch 650b1cf): 1 proc 160K, 2 procs 240K, 3 procs 285K, 4 procs 311K
hash/s aggregate (1.94x; VRAM caps 4 workers at m=28).

Start a scoped MPS daemon, run, and stop it:

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/flock-mps-pipe \\
           CUDA_MPS_LOG_DIRECTORY=/tmp/flock-mps-log
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d

    export CUDA_ROOT="$HOME/.local/cuda13-merged"
    export FRX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PATH="$HOME/.local/cuda13/bin:$PATH"
    PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \\
        python/flock_zorch/testing/prove_throughput_bench.py \\
        blake3 --golden blake3_ligerito_golden_m28.bin --procs 3

    echo quit | nvidia-cuda-mps-control

Each worker re-proves the same golden instance; throughput is what is measured,
and re-proving changes no kernel or shape. Workers warm up (compile) first,
align on a file barrier, then run their timed windows; the aggregate is the sum
of per-worker rates, honest only while all workers overlap — the printed window
skew states how well they did. VRAM bounds the worker count: ~7 GB per worker
at m=28 under `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

# ------------------------------------------------------------------ orchestrator
#
# The orchestrator imports no frx (it must not take the GPU); workers re-exec
# this file with --worker and do the proving. Keep the module import-light until
# worker_main so the orchestrator stays a pure process manager.

_WORKER_LINE = re.compile(
    r"WORKER tag=(?P<tag>\S+) m=(?P<m>\d+) n_hash=(?P<n_hash>\d+) "
    r"iters=(?P<iters>\d+) start=(?P<start>[\d.]+) end=(?P<end>[\d.]+)"
)


def _card_state() -> tuple[str, int]:
    """`(card state for the record, compute-process count)`; count is -1 when
    the probe fails, which never blocks. The frx-side `gpu_provenance` twin is
    not reused here so the orchestrator never imports frx — workers own the GPU."""
    try:
        used, total, util = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
            check=True).stdout.split("\n")[0].split(",")
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
        n = len([ln for ln in apps.splitlines() if ln.strip()])
        return f"{used.strip()}/{total.strip()} MiB, util {util.strip()}%", n
    except Exception as e:  # no nvidia-smi / output drift — record, don't block
        return f"state unknown ({type(e).__name__})", -1


def _mps_alive() -> bool:
    """True if an MPS control daemon answers on the ambient pipe directory."""
    try:
        out = subprocess.run(
            ["nvidia-cuda-mps-control"], input="get_server_list\n",
            capture_output=True, text=True, timeout=10)
        return out.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def orchestrate(args) -> int:
    if not _mps_alive() and not args.allow_no_mps:
        print(
            "REFUSING: no CUDA MPS control daemon reachable. Without MPS, "
            "concurrent prover processes time-slice contexts and aggregate "
            "BELOW one process (~0.9x measured) — the co-scheduling this bench "
            "exists to measure needs MPS. Start one (see module docstring) or "
            "pass --allow-no-mps to record the contended number anyway.",
            file=sys.stderr)
        return 2

    card, others = _card_state()
    # The MPS control/server daemons may appear as compute apps; more than that
    # means a real neighbour and the aggregate would be meaningless.
    if others > 2 and not args.allow_contended:
        print(f"REFUSING: {others} compute processes already on the card "
              f"({card}).", file=sys.stderr)
        return 2
    print(f"gpu: {card} | procs={args.procs} iters={args.iters} "
          f"golden={args.golden or '(default)'}")

    with tempfile.TemporaryDirectory(prefix="flock-throughput-") as barrier:
        cmd_base = [sys.executable, os.path.abspath(__file__), args.circuit,
                    "--worker", "--barrier", barrier,
                    "--procs", str(args.procs), "--iters", str(args.iters),
                    "--warmup", str(args.warmup)]
        if args.golden:
            cmd_base += ["--golden", args.golden]
        procs = [subprocess.Popen(cmd_base + ["--tag", f"w{i}"],
                                  stdout=subprocess.PIPE, text=True)
                 for i in range(args.procs)]
        outs = [p.communicate()[0] for p in procs]
        if any(p.returncode != 0 for p in procs):
            for p, out in zip(procs, outs):
                if p.returncode != 0:
                    print(f"worker exited {p.returncode}:\n{out}", file=sys.stderr)
            return 1

    rows = []
    for out in outs:
        match = _WORKER_LINE.search(out)
        if match is None:
            print(f"unparsable worker output:\n{out}", file=sys.stderr)
            return 1
        rows.append({k: float(v) if "." in v else int(v)
                     for k, v in match.groupdict().items() if k != "tag"}
                    | {"tag": match["tag"]})

    n_hash = rows[0]["n_hash"]
    rates = [r["iters"] * n_hash / (r["end"] - r["start"]) for r in rows]
    for r, rate in zip(rows, rates):
        print(f"  {r['tag']}: {r['iters']} proves in "
              f"{(r['end'] - r['start']) * 1e3:.0f}ms -> {rate:.0f} hash/s")

    # The sum of rates is honest only while every worker is running; the skew
    # says how much of the union window was NOT fully overlapped.
    union = max(r["end"] for r in rows) - min(r["start"] for r in rows)
    common = min(r["end"] for r in rows) - max(r["start"] for r in rows)
    skew = 100.0 * (1.0 - common / union) if union > 0 else 0.0
    print(f"aggregate {sum(rates):.0f} hash/s over {args.procs} procs "
          f"(m={rows[0]['m']}, {n_hash} hashes/proof; window skew {skew:.0f}%)")
    return 0


# ---------------------------------------------------------------------- worker


def worker_main(args) -> int:
    import frx

    frx.config.update("jax_enable_x64", True)

    from flock_zorch.testing._util import await_all
    from flock_zorch.testing.prove_phase_bench import Circuit, make_prove

    circ = Circuit(args.circuit)
    g = circ.ingest(args.golden)
    n_hash = circ.hashes_per_proof(g["meta"])
    prove = make_prove(circ, g, unpacked=False)

    for _ in range(args.warmup):
        await_all(prove())

    # File barrier: every worker warms (compiles) before any timed window opens.
    open(os.path.join(args.barrier, f"ready.{args.tag}"), "w").close()
    while len(os.listdir(args.barrier)) < args.procs:
        time.sleep(0.05)

    start = time.perf_counter()  # CLOCK_MONOTONIC: comparable across processes
    for _ in range(args.iters):
        await_all(prove())
    end = time.perf_counter()

    print(f"WORKER tag={args.tag} m={g['meta']['m']} n_hash={n_hash} "
          f"iters={args.iters} start={start:.6f} end={end:.6f}", flush=True)
    return 0


# ------------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("circuit", nargs="?", default="blake3",
                    choices=["blake3", "sha2", "keccak3"])
    ap.add_argument("--golden", help="golden filename under artifacts/, for "
                                     "m-variant dumps")
    ap.add_argument("--procs", type=int, default=3,
                    help="concurrent prover processes (VRAM-bounded: ~7 GB "
                         "per worker at m=28)")
    ap.add_argument("--iters", type=int, default=10,
                    help="timed proves per worker")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--allow-no-mps", action="store_true",
                    help="run without an MPS daemon (records the time-sliced "
                         "contended number, ~0.9x of one process)")
    ap.add_argument("--allow-contended", action="store_true",
                    help="measure even with other compute processes on the card")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--barrier", help=argparse.SUPPRESS)
    ap.add_argument("--tag", default="w0", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        return worker_main(args)
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
