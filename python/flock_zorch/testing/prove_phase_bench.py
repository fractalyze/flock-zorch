# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Per-phase GPU prove timing with a hashes/second column, across hash circuits.

The e2e_*_ligerito_bench scripts each report one wall-clock number for one
circuit. That is not enough to steer throughput work: it cannot say which of
commit / zerocheck / lincheck / open owns the time, and it never converts to the
metric the throughput goal is stated in. This harness does both, for every
Ligerito hash circuit, off the same goldens the byte gates use.

Three things it deliberately does that the per-circuit benches do not:

1. **Refuses to report absolute numbers on a contended GPU.** A neighbour
   process saturating the SMs inflates a warm prove by ~28x on this box (keccak3
   m=22 measured at 42 ms idle and 1171 ms against a 100%-utilisation
   neighbour). That is not the ~6% drift a concurrent *CPU* build causes, and it
   silently turns any absolute claim into fiction. `--allow-contended` overrides
   for ratio-only work, and marks the output.
2. **Reports hashes/second, not just milliseconds.** Each circuit packs
   `n_sub * 2^(m - k_log)` hashes into a proof, so cost per hash — the quantity
   a throughput target is about — differs from cost per proof by a
   circuit-dependent constant.
3. **Shows the segmentation cost.** Splitting phases requires awaiting each one,
   which serialises work the unsegmented prove may overlap. Both totals are
   printed; if they disagree materially, trust the unsegmented one for absolute
   claims and the split only for attribution.

Run:
    export CUDA_ROOT="$HOME/.local/cuda13-merged"
    export FRX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PATH="$HOME/.local/cuda13/bin:$PATH"
    PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" .venv/bin/python \\
        python/flock_zorch/testing/prove_phase_bench.py [circuit ...] [options]

    circuit         blake3 | keccak3 | sha2   (default: blake3)
    --golden NAME   golden under artifacts/ (default: the circuit's own)
    --runs N        timed iterations, best-of (default 3)
    --unpacked      send the witness as uint8 bits instead of packed F128
    --allow-contended   measure anyway, and label the output untrustworthy
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402

from flock_zorch import lincheck, prover, zerocheck  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402

PHASES = ("commit", "zerocheck", "lincheck", "open")


# ---------------------------------------------------------------- GPU contention

@dataclass(frozen=True)
class GpuState:
    util: int                  # -1 when the probe failed
    detail: str
    pids: tuple[int, ...]      # foreign compute PIDs (this process excluded)

    @property
    def busy(self) -> bool:
        return self.util > 0

    @property
    def unknown(self) -> bool:
        return self.util < 0


def gpu_state() -> GpuState:
    """Sample the card's contention state.

    `utilization.gpu` is the field that matters and the one easy to miss: memory
    held by an *idle* neighbour costs headroom but not speed, while a neighbour
    at 100% costs an order of magnitude. Reading only `memory.used` cannot tell
    the two apart, which is why the PID list alone is not a verdict — an idle
    neighbour is a risk (it may wake mid-run, and it does eat VRAM), not a
    disqualification. `main` samples this before and after and compares.
    """
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout.strip()
        used, total, util = (int(x) for x in q.split(",")[:3])
    except Exception as e:  # no nvidia-smi, CPU run, parse drift — don't block
        return GpuState(-1, f"GPU state unknown ({type(e).__name__}); not gating", ())

    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout.strip()
        pids = tuple(int(ln.split(",")[0]) for ln in apps.splitlines()
                     if ln.strip() and int(ln.split(",")[0]) != _OWN_PID)
    except Exception:
        pids = ()

    who = f", {len(pids)} foreign proc" if pids else ""
    return GpuState(util, f"{used}/{total} MiB, util {util}%{who}", pids)


_OWN_PID = 0  # set in main(); 0 never matches a real pid, so pre-set is safe


# ------------------------------------------------------------------- circuits

@dataclass(frozen=True)
class Circuit:
    name: str
    golden: str
    domain: bytes
    n_sub: int          # hashes packed per 2^k_log block
    load: Callable      # (golden) -> golden dict
    build: Callable     # (g, meta) -> lincheck circuit
    dump: str           # cargo example that regenerates the golden

    def ingest(self, golden: str | None):
        """Load a golden, or say how to make one. `artifacts/` is gitignored and
        the goldens are dumped on demand, so a missing file is the normal state
        on a fresh checkout, not a broken one."""
        name = golden or self.golden
        try:
            return self.load(name)
        except FileNotFoundError:
            raise SystemExit(
                f"missing golden artifacts/{name}\n"
                f"  regenerate with: cargo run --release --example {self.dump} -- "
                f"<n_hashes> \"$PWD/artifacts/{name}\"") from None


def _blake3():
    from flock_zorch.testing.blake3_ligerito_oracle_test import load
    return Circuit("blake3", "blake3_ligerito_golden.bin", b"flock-blake3-lig-v0", 1, load,
                   lambda g, meta: lincheck.CscCircuit(
                       g["a0_rows"], g["b0_rows"], 1 << meta["k_log"],
                       const_pin=meta["const_pin"]),
                   "dump_blake3_ligerito")


def _sha2():
    from flock_zorch.testing.sha2_ligerito_oracle_test import load
    return Circuit("sha2", "sha2_ligerito_golden.bin", b"flock-sha2-lig-v0", 1, load,
                   lambda g, meta: lincheck.CscCircuit(
                       g["a0_rows"], g["b0_rows"], 1 << meta["k_log"],
                       const_pin=meta["const_pin"]),
                   "dump_sha2_ligerito")


def _keccak3():
    from flock_zorch.lincheck.keccak3 import Keccak3LincheckCircuit
    from flock_zorch.testing.keccak3_ligerito_oracle_test import load
    # 3 independent Keccak-f[1600] permutations share one 2^17 block.
    return Circuit("keccak3", "keccak3_ligerito_golden.bin", b"flock-keccak3-lig-v0", 3, load,
                   lambda g, meta: Keccak3LincheckCircuit(),
                   "dump_keccak3_ligerito")


CIRCUITS = {"blake3": _blake3, "sha2": _sha2, "keccak3": _keccak3}


def hashes_per_proof(meta, n_sub: int) -> int:
    """Capacity, not occupancy: one proof commits 2^m bits laid out as
    2^(m - k_log) blocks of n_sub hashes each. A golden dumped with fewer hashes
    than that still pays the full proof cost, so throughput derived from a
    partly-filled golden understates the circuit (the keccak3 m=22 golden holds
    49 of 96 slots)."""
    return n_sub << (meta["m"] - meta["k_log"])


# -------------------------------------------------------------------- timing

def _await(x):
    frx.block_until_ready(frx.tree_util.tree_leaves(x))
    return x


def _unpack_bits(packed, m):
    packed = np.asarray(packed, np.uint64).reshape(-1, 2)
    bi = np.arange(64, dtype=np.uint64)
    lo = ((packed[:, 0:1] >> bi) & np.uint64(1)).astype(np.uint8)
    hi = ((packed[:, 1:2] >> bi) & np.uint64(1)).astype(np.uint8)
    return np.concatenate([lo, hi], axis=1).reshape(-1)[: 1 << m]


def make_prove(circ: Circuit, g, unpacked: bool):
    """Returns (prove_unsegmented, prove_segmented). Both run the identical
    sequence; the segmented one awaits between phases to attribute the time."""
    meta, cfg = g["meta"], g["cfg"]
    m, k_log, k_skip = meta["m"], meta["k_log"], meta["k_skip"]
    ir = k_log - k_skip
    z, stmt, zlc = g["z"], g["stmt"], g["zlc"]
    circuit = circ.build(g, meta)

    if unpacked:
        a_bits, b_bits, c_bits = (_unpack_bits(g["a"], m), _unpack_bits(g["b"], m),
                                  _unpack_bits(g["z"], m))
    else:
        # Packed F128 — witness_to_rows unpacks on device (8x less host transfer).
        a_bits, b_bits, c_bits = g["a"], g["b"], g["z"]

    def steps():
        """The prove, yielded phase by phase so both drivers share one definition
        and cannot drift apart."""
        root, pdata = yield "commit", lambda: zorch_ligerito.commit_flock_ligerito(cfg, z)
        ch = Challenger(circ.domain)
        prover.bind_statement(ch, stmt, root)
        zc = yield "zerocheck", lambda: zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch)
        x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
        lc = yield "lincheck", lambda: lincheck.prove(
            zlc, None, None, x_ab, m, k_log, k_skip, ch=ch, capture=True, circuit=circuit)
        lcc = lc[2]
        ab = fnp.concatenate([lcc.r_inner_rest, x_ab.x_outer], axis=0)
        cc = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
        yield "open", lambda: prover.open_batch_ligerito(cfg, z, pdata, [ab, cc], ch)

    def run(segmented: bool):
        times = {}
        gen = steps()
        send = None
        while True:
            try:
                name, fn = gen.send(send)
            except StopIteration:
                break
            if segmented:
                t0 = time.perf_counter()
                send = _await(fn())
                times[name] = (time.perf_counter() - t0) * 1e3
            else:
                send = fn()
        return times, send

    def unsegmented():
        _, last = run(segmented=False)
        return _await(last)

    def segmented():
        times, last = run(segmented=True)
        _await(last)
        return times

    return unsegmented, segmented


def best_ms(fn, runs: int) -> float:
    fn()  # warmup: compile + first transfer excluded
    return min(_timed(fn) for _ in range(runs))


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3


# ---------------------------------------------------------------------- main

def main() -> int:
    global _OWN_PID
    _OWN_PID = os.getpid()

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("circuits", nargs="*", default=["blake3"], choices=list(CIRCUITS))
    ap.add_argument("--golden", default=None, help="golden filename under artifacts/")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--unpacked", action="store_true",
                    help="send witness as uint8 bits (8x host transfer) instead of packed F128")
    ap.add_argument("--allow-contended", action="store_true",
                    help="measure even if the GPU is busy; output is labelled untrustworthy")
    args = ap.parse_args()

    before = gpu_state()
    if before.busy and not args.allow_contended:
        print(f"REFUSING to measure: {before.detail}", file=sys.stderr)
        print("A neighbour saturating the SMs inflates a warm prove by ~28x on this box, "
              "so absolute numbers would be fiction. Wait for the card, or pass "
              "--allow-contended for ratio-only work.", file=sys.stderr)
        return 2

    banner = ("CONTENDED — RATIOS ONLY, ABSOLUTE NUMBERS INVALID" if before.busy
              else "unknown GPU state" if before.unknown else "clean")
    print(f"device {frx.devices()[0]} | gpu: {before.detail} | {banner}")
    if before.pids and not before.busy:
        # Idle now is not idle later: a neighbour mid-compile shows 0% and then
        # takes the SMs. The after-check below is what actually catches it.
        print(f"  note: {len(before.pids)} foreign process(es) idle on the card — "
              "they may wake mid-run; the post-run check will say if they did.")
    print(f"witness form: {'uint8 bits' if args.unpacked else 'packed F128'} | best-of-{args.runs}\n")

    hdr = (f"{'circuit':>8} {'m':>3} {'hashes':>8} " +
           " ".join(f"{p:>10}" for p in PHASES) +
           f" {'Σphases':>9} {'prove':>9} {'hash/s':>10}")
    print(hdr)
    print("-" * len(hdr))

    for name in args.circuits:
        circ = CIRCUITS[name]()
        g = circ.ingest(args.golden)
        meta = g["meta"]
        n_hash = hashes_per_proof(meta, circ.n_sub)

        unseg, seg = make_prove(circ, g, args.unpacked)
        total = best_ms(unseg, args.runs)
        seg()  # warm the segmented driver too
        splits = [seg() for _ in range(args.runs)]
        # Attribute with the run whose total is lowest — same "best-of" rule as
        # the headline number, so the split describes that run and not an average
        # smeared across scheduler noise.
        parts = min(splits, key=lambda d: sum(d.values()))
        ssum = sum(parts.values())

        print(f"{name:>8} {meta['m']:>3} {n_hash:>8} " +
              " ".join(f"{parts[p]:>9.2f}ms" for p in PHASES) +
              f" {ssum:>7.1f}ms {total:>7.1f}ms {n_hash / (total / 1e3):>10.0f}")

        share = "  " + "  ".join(f"{p} {100 * parts[p] / ssum:.0f}%" for p in PHASES)
        print(share)
        if abs(ssum - total) / total > 0.10:
            print(f"  NOTE segmentation cost {ssum - total:+.1f}ms ({100 * (ssum - total) / total:+.0f}%) — "
                  "phases overlap in the unsegmented prove; use Σphases for attribution only.")

    # A card that was quiet at the start can be taken over mid-run, which is the
    # failure mode the pre-check cannot see. Say so rather than emitting a number
    # that silently carries someone else's load.
    after = gpu_state()
    if not before.unknown and (after.busy or set(after.pids) - set(before.pids)):
        print(f"\nWARNING: GPU state changed during the run "
              f"(before: {before.detail} | after: {after.detail}).\n"
              "Treat the numbers above as contended and re-measure.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
