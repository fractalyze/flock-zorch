# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Per-phase GPU prove timing with a hashes/second column, across hash circuits.

The e2e_*_ligerito_bench scripts each report one wall-clock number for one
circuit. That is not enough to steer throughput work: it cannot say which of
commit / zerocheck / lincheck / open owns the time, and it never converts to the
metric a throughput goal is stated in. This does both, for every Ligerito hash
circuit, off the goldens the byte gates already use.

It **reports hashes/second**: each circuit packs `n_sub * 2^(m - k_log)` hashes
into a proof, so cost per hash — the quantity a throughput target is about —
differs from cost per proof by a circuit-dependent constant.

It also refuses to run while another compute process holds the GPU, since a
neighbour saturating the SMs inflates a warm prove by ~28x on this box. That is
a precondition check, not a certificate: see the GPU-provenance section below
for what it cannot see.

**One run of this is not a baseline.** The best-of-N below is within a single
process. Measured on an idle card, blake3 landed 13-19% apart *across*
processes at m <= 26, almost all of it inside `open`, which falls into distinct
clusters run to run while `zerocheck` reproduces to 2.5%. It is not thermal —
a back-to-back batch held 41-47C at pegged clocks with no throttle reason, and
wall time did not track temperature. So take several runs, report the spread,
and treat a single number at m <= 26 as having a wide error bar.

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


# ---------------------------------------------------------------- GPU provenance
#
# What this records, and what it deliberately does not claim.
#
# It writes down what the card looked like, and refuses on one unambiguous
# fact: another compute process is on it. It does **not** certify that a
# measurement is trustworthy, and no output here should be read as doing so.
# It cannot see a host-side stall (which inflates every phase at once — the
# signature is all four moving together), non-compute graphics load, a
# neighbour that starts and exits between two samples, or the clock and
# thermal state.
#
# It also cannot see the largest source of error. On this box the same binary
# measured 13-19% apart across processes on an idle card at m <= 26, almost
# entirely inside `open` — coarser than most regressions worth benchmarking.
# A free card is a precondition for measuring, not evidence that a number is
# good; that comes from repeating the run and reporting the spread.


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


def gpu_provenance() -> tuple[str, int]:
    """`(card state for the record, count of other compute processes)`.

    The count is `-1` when the probe fails — no nvidia-smi, a CPU run, output
    drift — which never blocks a measurement.

    `memory.used` and `utilization.gpu` are card-wide and include this process
    (nvidia-smi attributes neither per process, and importing frx has already
    taken a context by the time anything here runs), so they are recorded and
    never compared against a threshold. Only the compute-app list names *other*
    processes, so it is the one thing worth acting on.
    """
    gpu = _visible_gpu()
    try:
        # One row per GPU. Aggregate rather than taking row 0: frx's device 0
        # need not be nvidia-smi's, and watching the wrong card silently is
        # worse than being occasionally too conservative.
        rows = [r.split(",") for r in
                _smi("gpu=memory.used,memory.total,utilization.gpu", gpu).splitlines()
                if r.strip()]
        used = sum(int(r[0]) for r in rows)
        total = sum(int(r[1]) for r in rows)
        util = max(int(r[2]) for r in rows)
    except Exception as e:
        return f"state unknown ({type(e).__name__})", -1

    try:
        own = os.getpid()
        others = [p for p in (int(ln.split(",")[0])
                              for ln in _smi("compute-apps=pid,used_memory",
                                             gpu).splitlines() if ln.strip())
                  if p != own]
    except Exception:
        return f"{used}/{total} MiB, util {util}% (compute-app list unavailable)", -1

    who = f", {len(others)} other compute proc" if others else ", no other compute proc"
    return f"{used}/{total} MiB, util {util}%{who}", len(others)


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
                                ch=ch, circuit=circuit)
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
                    help="measure even with another compute process on the card")
    args = ap.parse_args()

    if len(args.circuits) > 1:
        for flag, val in (("--golden", args.golden), ("--cpu-ms", args.cpu_ms)):
            if val is not None:
                ap.error(f"{flag} describes one instance; pass a single circuit with it")

    card, others = gpu_provenance()
    if others > 0 and not args.allow_contended:
        print(f"REFUSING to measure: {others} other compute process(es) on the card "
              f"({card}).\nA neighbour saturating the SMs inflates a warm prove by "
              "~28x on this box. Wait for the card, or pass --allow-contended for "
              "ratio-only work.", file=sys.stderr)
        return 2

    print(f"device {frx.devices()[0]} | gpu: {card}")
    print(f"witness form: {'uint8 bits' if args.unpacked else 'packed F128'} "
          f"| best-of-{args.runs} within this process\n")

    hdr = (f"{'circuit':>8} {'m':>3} {'hashes':>8} " +
           " ".join(f"{p:>10}" for p in PHASES) +
           f" {'sum':>9} {'wall':>9} {'hash/s':>10}")
    print(hdr)
    print("-" * len(hdr))

    for name in args.circuits:
        bench(Circuit(name), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
