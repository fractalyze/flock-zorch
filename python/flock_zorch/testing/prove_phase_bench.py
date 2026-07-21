# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Per-phase GPU prove timing with a hashes/second column, across hash circuits.

The e2e_*_ligerito_bench scripts each report one wall-clock number for one
circuit. That is not enough to steer throughput work: it cannot say which of
commit / zerocheck / lincheck / open owns the time, and it never converts to the
metric a throughput goal is stated in. This does both, for every Ligerito hash
circuit, off the goldens the byte gates already use.

Two things it deliberately does that the per-circuit benches do not:

1. **Refuses to report absolute numbers on a contended GPU.** A neighbour
   saturating the SMs inflates a warm prove by ~28x on this box. That is not the
   ~6% drift a concurrent *CPU* build causes, and it silently turns any absolute
   claim into fiction.
2. **Reports hashes/second.** Each circuit packs `n_sub * 2^(m - k_log)` hashes
   into a proof, so cost per hash — the quantity a throughput target is about —
   differs from cost per proof by a circuit-dependent constant.

**Every phase is awaited before the next starts**, so the split accounts for the
whole prove and each phase is billed the work it actually causes. The cost is
that awaiting serialises work an un-instrumented prove may overlap, which makes
the reported total an *upper* bound on a real prove — the conservative direction
for a "how far are we from the target" question. (The commit->zerocheck boundary
already syncs regardless: commit ends by pulling the root to host.) `Sum` versus
`wall` is printed as a self-check on the instrumentation, not as a claim about
overlap: a gap there means work escaped every phase.

Run:
    export CUDA_ROOT="$HOME/.local/cuda13-merged"
    export FRX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PATH="$HOME/.local/cuda13/bin:$PATH"
    PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \\
        python/flock_zorch/testing/prove_phase_bench.py [circuit ...] [options]
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

import frx

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402

from flock_zorch import lincheck, prover, zerocheck  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.testing._golden import unpack_bits  # noqa: E402
from flock_zorch.testing._util import await_all, best_of  # noqa: E402

PHASES = ("commit", "zerocheck", "lincheck", "open")


# ---------------------------------------------------------------- GPU contention

@dataclass(frozen=True)
class GpuState:
    util: int | None           # None when the probe failed
    detail: str
    pids: frozenset[int]       # foreign compute PIDs (this process excluded)

    @property
    def busy(self) -> bool:
        # Importing frx initializes this process on the selected GPU before the
        # guard runs, so its own warm-up can account for nonzero utilization.
        # Contention requires activity plus another compute process.
        return self.util is not None and self.util > 0 and bool(self.pids)

    @property
    def unknown(self) -> bool:
        return self.util is None

    @property
    def label(self) -> str:
        if self.unknown:
            return "unknown GPU state"
        return "CONTENDED — ABSOLUTE NUMBERS INVALID" if self.busy else "clean"


def _visible_gpu() -> str | None:
    """Physical GPU selected by a single numeric CUDA_VISIBLE_DEVICES entry."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return visible if visible.isdigit() else None


def _smi(query: str, gpu: str | None = None) -> str:
    cmd = ["nvidia-smi"]
    if gpu is not None:
        cmd += ["-i", gpu]
    cmd += [f"--query-{query}", "--format=csv,noheader,nounits"]
    return subprocess.run(cmd,
                          capture_output=True, text=True, timeout=15,
                          check=True).stdout.strip()


def gpu_state() -> GpuState:
    """Sample the card's contention state.

    `utilization.gpu` is the field that matters and the one easy to miss: memory
    held by an *idle* neighbour costs headroom but not speed, while a neighbour
    at 100% costs an order of magnitude. Reading only `memory.used` cannot tell
    the two apart, which is why the PID list alone is not a verdict — an idle
    neighbour is a risk (it may wake mid-run, and it does eat VRAM), not a
    disqualification. Callers sample before and after and compare.
    """
    try:
        # CUDA_VISIBLE_DEVICES renumbers frx's devices.  When it names one
        # physical numeric GPU, query that card; otherwise conservatively
        # aggregate all cards because frx device 0 need not be nvidia-smi's 0.
        gpu = _visible_gpu()
        rows = [r.split(",") for r in
                _smi("gpu=memory.used,memory.total,utilization.gpu", gpu).splitlines()
                if r.strip()]
        used = sum(int(r[0]) for r in rows)
        total = sum(int(r[1]) for r in rows)
        util = max(int(r[2]) for r in rows)
    except Exception as e:  # no nvidia-smi, CPU run, parse drift — don't block
        return GpuState(None, f"GPU state unknown ({type(e).__name__}); not gating",
                        frozenset())

    pids = foreign_pids(gpu)
    who = f", {len(pids)} foreign proc" if pids else ""
    return GpuState(util, f"{used}/{total} MiB, util {util}%{who}", pids)


def foreign_pids(gpu: str | None = None) -> frozenset[int]:
    """Compute PIDs on the card other than our own. Empty when the probe fails."""
    try:
        own = os.getpid()
        return frozenset(
            p for p in (int(ln.split(",")[0])
                        for ln in _smi("compute-apps=pid,used_memory", gpu).splitlines()
                        if ln.strip())
            if p != own)
    except Exception:
        return frozenset()


class NeighbourWatch:
    """Poll for foreign compute PIDs *while* a row is measured.

    The sample taken before a row and the one taken after it are a whole
    measurement apart, and a neighbour that starts and exits inside that window
    leaves no trace in either — the row comes back inflated and is reported
    clean. Not hypothetical: an m=28 blake3 row measured 179.4 ms against a
    106.0 ms best, 67% slow, with both endpoint samples showing an empty card.

    Utilization cannot be the signal here — during a measurement the card reads
    busy because of *us* — but `compute-apps` lists only other processes, so
    presence is. Each tick is a subprocess, so the interval trades detection
    against the host noise it adds to the very thing being timed; a neighbour
    living entirely inside one tick is still missed. The watch spans the whole
    row — golden ingest and warmup included, not just the timed runs — which is
    the conservative direction: it can ask for a re-measure on a neighbour that
    never overlapped a timed prove.
    """

    def __init__(self, interval: float = 0.25) -> None:
        self._interval = interval
        self._gpu = _visible_gpu()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.seen: frozenset[int] = frozenset()

    def _poll(self) -> None:
        # Sample first, then wait: a row shorter than one interval must still
        # get a look, and the gap between the pre-row sample and the start of
        # the measurement (the golden ingest, seconds for the larger dumps) is
        # otherwise unwatched.
        while True:
            self.seen |= foreign_pids(self._gpu)
            if self._stop.wait(self._interval):
                return

    def __enter__(self) -> NeighbourWatch:
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def contended_after(before: GpuState, after: GpuState,
                    seen: frozenset[int] = frozenset()) -> bool:
    """Did a neighbour take the card while we were measuring?

    `seen` is every foreign PID NeighbourWatch caught mid-row. A PID in there
    but in neither endpoint sample is the come-and-go case the endpoints cannot
    see, and it is the whole reason the watch exists.

    `after.busy` covers the other direction — a neighbour that was already
    there when we started and is on the SMs now. It needs no separate PID check
    because `busy` already requires one.
    """
    if before.unknown:
        return False
    arrived = (after.pids | seen) - before.pids
    return bool(arrived) or after.busy


# ------------------------------------------------------------------- circuits

def _csc(g):
    meta = g["meta"]
    return lincheck.CscCircuit(g["a0_rows"], g["b0_rows"], 1 << meta["k_log"],
                               const_pin=meta["const_pin"])


def _keccak3_circuit(_g):
    from flock_zorch.lincheck.keccak3 import Keccak3LincheckCircuit
    return Keccak3LincheckCircuit()


# name -> lincheck circuit builder. Everything else — golden filename, FS domain,
# dump example, loader, unpacker — follows from the name by the repo's own
# naming, so there is one place per circuit to get wrong.
CIRCUITS: dict[str, Callable] = {
    "blake3": _csc, "sha2": _csc, "keccak3": _keccak3_circuit,
}


@dataclass(frozen=True)
class Circuit:
    name: str

    @property
    def golden(self) -> str:
        return f"{self.name}_ligerito_golden.bin"

    @property
    def domain(self) -> bytes:
        return f"flock-{self.name}-lig-v0".encode()

    @property
    def n_sub(self) -> int:
        """Hashes packed into one 2^k_log block."""
        if self.name == "keccak3":
            from flock_zorch.lincheck.keccak3 import N_SUB
            return N_SUB  # 3 independent Keccak-f[1600] permutations per block
        return 1

    def build(self, g):
        return CIRCUITS[self.name](g)

    @property
    def _oracle(self):
        return importlib.import_module(
            f"flock_zorch.testing.{self.name}_ligerito_oracle_test")

    def ingest(self, golden: str | None):
        """Load a golden through the gate's own loader, so the bench and the byte
        gate can never disagree about the wire. A missing file is reported by
        `_golden.open_golden`."""
        return self._oracle.load(golden or self.golden)

    def hashes_per_proof(self, meta) -> int:
        """Capacity, not occupancy: one proof commits 2^m bits laid out as
        2^(m - k_log) blocks of n_sub hashes each. A golden dumped with fewer
        hashes than that still pays the full proof cost, so throughput derived
        from a partly-filled golden understates the circuit (the keccak3 m=22
        golden holds 49 of 96 slots)."""
        return self.n_sub << (meta["m"] - meta["k_log"])


# -------------------------------------------------------------------- timing

def make_prove(circ: Circuit, g, unpacked: bool):
    """Returns a `prove(times) -> result` running one full prove.

    With `times`, every phase is awaited and recorded into it. There is exactly
    one definition of the sequence, and every statement lives inside a phase —
    inter-phase glue billed to nobody would make the split silently under-count
    the prove it claims to decompose.
    """
    meta, cfg = g["meta"], g["cfg"]
    m, k_log, k_skip = meta["m"], meta["k_log"], meta["k_skip"]
    ir = k_log - k_skip
    stmt, zlc = g["stmt"], g["zlc"]
    circuit = circ.build(g)

    if unpacked:
        witness = (unpack_bits(g["a"], m), unpack_bits(g["b"], m),
                   unpack_bits(g["z"], m))
    else:
        # Packed F128 — witness_to_rows unpacks on device (8x less host transfer).
        witness = (g["a"], g["b"], g["z"])
    # Upload once. Left as host numpy these re-cross PCIe every iteration, and the
    # cost lands on whichever phase touches them first — skewing the very split
    # this harness exists to report.
    a_bits, b_bits, c_bits = (frx.device_put(x) for x in witness)
    z = frx.device_put(g["z"])

    def prove(times=None):
        def phase(name, fn):
            if times is None:
                return fn()
            t0 = time.perf_counter()
            r = await_all(fn())
            times[name] = (time.perf_counter() - t0) * 1e3
            return r

        def _commit():
            root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, z)
            ch = Challenger(circ.domain)
            prover.bind_statement(ch, stmt, root)
            return pdata, ch

        def _lincheck(zc):
            x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
            lc = lincheck.prove(zlc, None, None, x_ab, m, k_log, k_skip,
                                ch=ch, capture=True, circuit=circuit)
            return x_ab, lc

        def _open(zc, x_ab, lc):
            ab = fnp.concatenate([lc.claim.r_inner_rest, x_ab.x_outer], axis=0)
            cc = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
            return prover.open_batch_ligerito(cfg, z, pdata, [ab, cc], ch)

        pdata, ch = phase("commit", _commit)
        zc = phase("zerocheck",
                   lambda: zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch))
        x_ab, lc = phase("lincheck", lambda: _lincheck(zc))
        return phase("open", lambda: _open(zc, x_ab, lc))

    return prove


# ---------------------------------------------------------------------- main

def bench(circ: Circuit, args) -> None:
    """Measure one circuit and print its row. Scoped to a function so the golden
    (~90 MB) and the circuit's device buffers are released before the next one."""
    g = circ.ingest(args.golden)
    meta = g["meta"]
    n_hash = circ.hashes_per_proof(meta)
    prove = make_prove(circ, g, args.unpacked)

    def timed_prove():
        times = {}
        return prove(times), times

    wall, parts = best_of(timed_prove, args.runs)
    total = sum(parts.values())

    print(f"{circ.name:>8} {meta['m']:>3} {n_hash:>8} " +
          " ".join(f"{parts[p]:>9.2f}ms" for p in PHASES) +
          f" {total:>7.1f}ms {wall:>7.1f}ms {n_hash * 1e3 / wall:>10.0f}")
    print("  " + "  ".join(f"{p} {100 * parts[p] / total:.0f}%" for p in PHASES))
    if args.cpu_ms:
        print(f"  {args.cpu_ms / wall:.2f}x vs same-instance flock CPU {args.cpu_ms:.0f}ms")
    if abs(total - wall) / wall > 0.10:
        print(f"  NOTE {wall - total:+.1f}ms ({100 * (wall - total) / wall:+.0f}%) of the "
              "prove is outside every phase — the split under-counts; instrumentation bug.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("circuits", nargs="*", default=["blake3"], choices=list(CIRCUITS))
    ap.add_argument("--golden", help="golden filename under artifacts/, for m-variant "
                                     "dumps (single circuit only)")
    ap.add_argument("--runs", type=int, default=3, help="timed iterations, best-of")
    ap.add_argument("--cpu-ms", type=float, help="flock CPU ms for the same instance "
                                                 "(from bench_<circuit>_cpu), to print a "
                                                 "speedup; single circuit only")
    ap.add_argument("--unpacked", action="store_true",
                    help="send witness as uint8 bits (8x host transfer) not packed F128")
    ap.add_argument("--allow-contended", action="store_true",
                    help="measure even if the GPU is busy; output is labelled untrustworthy")
    args = ap.parse_args()

    if len(args.circuits) > 1:
        for flag, val in (("--golden", args.golden), ("--cpu-ms", args.cpu_ms)):
            if val is not None:
                ap.error(f"{flag} describes one instance; pass a single circuit with it")

    before = gpu_state()
    if before.busy and not args.allow_contended:
        print(f"REFUSING to measure: {before.detail}\n"
              "A neighbour saturating the SMs inflates a warm prove by ~28x on this box, "
              "so absolute numbers would be fiction. Wait for the card, or pass "
              "--allow-contended for ratio-only work.", file=sys.stderr)
        return 2

    print(f"device {frx.devices()[0]} | gpu: {before.detail} | {before.label}")
    if before.pids and not before.busy:
        # Idle now is not idle later: a neighbour mid-compile shows 0% and then
        # takes the SMs. The per-circuit check below is what actually catches it.
        print(f"  note: {len(before.pids)} foreign process(es) idle on the card — "
              "they may wake mid-run; each row is re-checked after it is measured.")
    print(f"witness form: {'uint8 bits' if args.unpacked else 'packed F128'} "
          f"| best-of-{args.runs}\n")

    hdr = (f"{'circuit':>8} {'m':>3} {'hashes':>8} " +
           " ".join(f"{p:>10}" for p in PHASES) +
           f" {'sum':>9} {'wall':>9} {'hash/s':>10}")
    print(hdr)
    print("-" * len(hdr))

    dirty = False
    for name in args.circuits:
        # The watch lives here rather than inside bench() so that sampling the
        # card, measuring, and judging the row are all owned by one scope — the
        # verdict needs `before`, `after` and `seen` together.
        with NeighbourWatch() as watch:
            bench(Circuit(name), args)
        # Re-check per row, not once at the end: on a multi-circuit run a mid-run
        # takeover would otherwise invalidate every row without saying which.
        after = gpu_state()
        if contended_after(before, after, watch.seen):
            print(f"  WARNING contended while measuring (now: {after.detail}) — "
                  "treat this row as invalid and re-measure.", file=sys.stderr)
            dirty = True
    return 1 if dirty else 0


if __name__ == "__main__":
    raise SystemExit(main())
