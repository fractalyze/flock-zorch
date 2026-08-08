# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Isolate ONE zerocheck prover round for profiling, so a capture contains that
round's kernels and nothing else.

`prove_phase_bench.py` answers "which phase owns the time". This answers the
next question down — "which kernels inside one round own it" — which
`docs/measurement.md` insists on before work is scoped ("a phase is not a
target": zerocheck bundles the round-1 URM and the multilinear ladder, and the
two have turned out to have *opposite* bindings, the URM arithmetic-bound and
the ladder bandwidth-bound).

Why this is a committed file rather than a scratch script: the same harness had
been rebuilt from scratch in several consecutive sessions. It encodes method
that is not re-derivable from the code —

* **`cuProfilerStart`/`cuProfilerStop` gating.** Bracketing the window excludes
  warm-up and autotune *by construction*, instead of trying to filter them out
  of a whole-process capture afterwards. That is what makes a bucket table
  trustworthy.
* **`--cuda-graph-trace=node` is mandatory.** Without it XLA's CUDA-graph
  dispatches under-report kernels by ~50×.
* **Walls never come from under nsys.** It inflates *host* dispatch ~2×;
  on-device kernel durations are unaffected. So: kernel times from the capture,
  walls from `--mode walls` or a clean un-profiled run.
* **Per-iteration state must be rebuilt.** The transcript is threaded, so
  reusing state across iterations measures a transcript another iteration
  already advanced.

The round objects come from the public `zerocheck_steps`, and `prove_rounds` is
a bare loop over `rnd(carry, transcript)` — so driving the rounds one at a time
here is the shipped path, not an approximation of it. The carry construction
below mirrors `zerocheck.prove_packed`; if that changes shape, this must follow.

Run (see `docs/measurement.md` — `ptxas` must be 13.3, and do NOT set
`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`):

    export CUDA_ROOT=<a 13.3 toolchain>
    export PATH="$CUDA_ROOT/bin:$PATH"
    export FRX_PLATFORMS=cuda,cpu FRX_ENABLE_X64=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    unset JAX_PLATFORMS JAX_ENABLE_X64
    PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \\
        python/flock_zorch/testing/round_window_bench.py --round ml --mode walls

    # kernel attribution: gate the window, then read the report
    nsys profile -t cuda --cuda-graph-trace=node \\
        --capture-range=cudaProfilerApi --capture-range-end=stop -o out \\
        <the same command> --mode capture
    nsys stats --report cuda_gpu_kern_sum out.nsys-rep
"""
from __future__ import annotations

import argparse
import ctypes
import statistics
import time

import frx

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402

from flock_zorch import lincheck, prover, sumcheck  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.testing._util import await_all  # noqa: E402
from flock_zorch.testing.prove_phase_bench import Circuit  # noqa: E402
from flock_zorch.zerocheck import prover as zcp  # noqa: E402
from flock_zorch.zerocheck._fold import _fold_at_z, _lagrange_weights  # noqa: E402

ROUNDS = ("urm", "ml")


class ProfilerGate:
    """`cuProfilerStart`/`cuProfilerStop` around the measured window.

    A missing `libcuda` degrades to a no-op rather than refusing to run, since
    `--mode walls` does not need the gate at all.
    """

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None
        try:
            self._lib = ctypes.CDLL("libcuda.so.1")
        except OSError:
            self._lib = None

    def __enter__(self):
        if self._lib is not None:
            self._lib.cuProfilerStart()
        return self

    def __exit__(self, *exc):
        if self._lib is not None:
            self._lib.cuProfilerStop()
        return False


def make_state(circ: Circuit, golden: str | None, upto: str):
    """Returns `fresh_state()`, giving the state the requested round starts from.

    `upto="urm"` stops before the URM; `upto="ml"` runs the URM first and awaits
    it, so it is fully settled *outside* the window the caller then measures.
    """
    g = circ.ingest(golden)
    meta, cfg = g["meta"], g["cfg"]
    m, k_log = meta["m"], meta["k_log"]
    a_bits, b_bits, c_bits = (frx.device_put(x) for x in (g["a"], g["b"], g["z"]))
    z = frx.device_put(g["z"])
    # Upload the lincheck stripe as the bench does: left on the host it would
    # re-cross PCIe and land on whichever round touches it first.
    lincheck.stripe_to_device(g["zlc"], m, k_log)

    def fresh_state():
        root, _pdata = zorch_ligerito.commit_flock_ligerito(cfg, z)
        ch = Challenger(circ.domain)
        prover.bind_statement(ch, g["stmt"], root)

        k_skip = zcp.K_SKIP
        r_skip, r_outer = zcp.sample_challenge_coords(ch, m, k_skip)
        r = fnp.concatenate([r_skip, zcp._SMALL_G, zcp._MEDIUM_G, r_outer])
        carry = zcp._ZerocheckCarry(a_bits, b_bits, c_bits, r=r)
        urm, ml = zcp.zerocheck_steps(m, k_skip)
        if upto == "ml":
            carry, ch, _ = urm(carry, ch)
            await_all(carry)
        return carry, ch, (ml if upto == "ml" else urm), k_skip

    return fresh_state, m


def ml_substeps(carry, ch, k_skip):
    """`_MultilinearRound.__call__` unrolled with an await between sub-steps.

    The awaits serialize work the real round may overlap, so these are upper
    bounds — the same caveat `prove_phase_bench`'s split carries. Their value is
    as a cross-check that the nsys kernel buckets add up to the right places.
    Kept deliberately close to the round body so the two can be diffed by eye.
    """
    t: dict[str, float] = {}

    def step(name, fn):
        t0 = time.perf_counter()
        r = await_all(fn())
        t[name] = (time.perf_counter() - t0) * 1e3
        return r

    weights = step("lagrange_weights", lambda: _lagrange_weights(k_skip, carry.z, 0))
    a_g = step("fold_at_z(a)", lambda: _fold_at_z(carry.a_rows, weights))
    b_g = step("fold_at_z(b)", lambda: _fold_at_z(carry.b_rows, weights))
    eq = step("EQ_TABLES", lambda: zcp._EQ_TABLES(carry.r[k_skip + 1 :]))
    step(
        "mlv_sumcheck",
        lambda: zcp._mlv_sumcheck(a_g, b_g, eq, sumcheck.eq._ONE_G, ch._t),
    )
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit", nargs="?", default="blake3")
    ap.add_argument("--golden", help="golden filename under artifacts/")
    ap.add_argument("--round", choices=ROUNDS, default="ml")
    ap.add_argument(
        "--mode",
        choices=("capture", "walls"),
        default="walls",
        help="capture: one gated window for nsys. walls: timed runs, plus "
        "sub-step buckets for --round ml",
    )
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    circ = Circuit(args.circuit)
    fresh_state, m = make_state(circ, args.golden, args.round)

    def run_window(state):
        carry, ch, rnd, _k_skip = state
        return rnd(carry, ch)

    await_all(run_window(fresh_state()))  # compile + autotune, outside every window

    if args.mode == "capture":
        state = fresh_state()
        with ProfilerGate():
            await_all(run_window(state))
        print(f"zc.{args.round} m={m}: capture done — kernel times are in the report")
        return 0

    walls, buckets = [], []
    for _ in range(args.runs):
        state = fresh_state()
        t0 = time.perf_counter()
        await_all(run_window(state))
        walls.append((time.perf_counter() - t0) * 1e3)
        if args.round == "ml":
            c, ch, _rnd, k_skip = fresh_state()
            buckets.append(ml_substeps(c, ch, k_skip))

    print(f"zc.{args.round} window, m={m}, {args.runs} runs")
    print(
        f"  wall ms: min {min(walls):.3f}  median {statistics.median(walls):.3f}  "
        f"max {max(walls):.3f}"
    )
    if buckets:
        print("  sub-step buckets (awaited, upper bounds; min across runs):")
        total = 0.0
        for name in buckets[0]:
            v = min(b[name] for b in buckets)
            total += v
            print(f"    {name:22s} {v:8.3f} ms")
        print(f"    {'sum':22s} {total:8.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
