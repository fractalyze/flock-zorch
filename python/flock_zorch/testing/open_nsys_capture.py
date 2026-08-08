# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""nsys capture + attribution harness for the `open` phase, and an open-only
wall sampler.

The open-phase perf work (the "attribute the Ligerito recursive open" board
item) needs three measurements this file provides in one place, because its
capture scaffolding was previously rebuilt from scratch in three separate
sessions:

- **capture** (default): run commit -> zerocheck -> lincheck once, warm the
  open, then replay ONE open iteration inside a `cudaProfilerApi` capture
  range with NVTX ranges around the open's four sub-steps
  (`ring_switch.prove_batched`, `_combine_claims`, `_open_jitted`,
  `_flock_proof_dict`). Run it UNDER nsys — see the recipe below.
- **--walls N**: the same open, replayed N times un-profiled, reporting
  p10 / median / min. This is the clean-wall channel; nsys inflates host
  dispatch ~2x, so wall numbers must never come from the capture run.
- **--report FILE.sqlite**: read an exported capture and print device-busy
  attribution: per sub-step, per CUDA kernel, per XLA HLO-op family, and per
  recursion level (segmented at the per-level query-sampler callbacks).

The open is replayable because `Challenger` advances by REPLACING its
immutable `_t` pytree: snapshotting the post-lincheck `_t` and re-assigning
it before each iteration replays byte-identical Fiat-Shamir draws, so every
iteration runs the same program on the same data.

Attribution method: `busy` = sum of on-device kernel durations, `wall` =
clean un-profiled runs, `latency` = wall - busy. Kernels are attributed to
NVTX ranges by HOST-ENQUEUE containment — each kernel's CUDA API row (joined
via correlationId) locates the dispatch inside the range that issued it.
This deliberately differs from the device-timestamp containment earlier
sessions used: dispatch is async, so device-time containment re-attributes an
early sub-step's kernels to whichever range the host had reached by the time
they ran (measured here: it moved ring_switch's bit-select reduces into the
jitted-open range), and it lands kernels that overlap a blocking
`pure_callback` inside the callback's thunk range. The enqueue anchor fixes
both.

Run (the toolchain preamble is load-bearing — ptxas 12.9 silently selects
soft-GHASH, 5.5x on the prove; `commit` at tens of ms instead of ~1.3 ms at
m28 is the tell):

    export CUDA_ROOT=$HOME/.local/cuda13   # must be a 13.3 ptxas
    export FRX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
    unset JAX_PLATFORMS XLA_PYTHON_CLIENT_ALLOCATOR JAX_COMPILATION_CACHE_DIR
    export PATH="$CUDA_ROOT/bin:$PATH"
    PY="python:$(scripts/zorch_pythonpath.sh)"

    # clean walls (no nsys):
    PYTHONPATH="$PY" .venv/bin/python \
        python/flock_zorch/testing/open_nsys_capture.py --golden \
        blake3_ligerito_golden_m32.bin --walls 12

    # capture:
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        --cuda-graph-trace=node --trace=cuda,nvtx -o open_m32 \
        --force-overwrite true env PYTHONPATH="$PY" .venv/bin/python \
        python/flock_zorch/testing/open_nsys_capture.py --golden \
        blake3_ligerito_golden_m32.bin
    nsys export --type sqlite --force-overwrite true -o open_m32.sqlite \
        open_m32.nsys-rep

    # report:
    .venv/bin/python python/flock_zorch/testing/open_nsys_capture.py \
        --report open_m32.sqlite
"""
from __future__ import annotations

import argparse
import ctypes
import sqlite3
import time
from collections import defaultdict

# ------------------------------------------------------------------ NVTX / CUPTI
#
# CUDA 13 dropped `libnvToolsExt.so.1`; the v3 C symbols live in
# `libnvtx3interop.so.1` (shipped under the CUDA toolkit's targets lib). The
# profiler start/stop pair comes from the driver directly.


def _load_nvtx():
    for name in ("libnvtx3interop.so.1", "libnvToolsExt.so.1"):
        for prefix in (
            "",
            "/usr/local/cuda/targets/x86_64-linux/lib/",
            "/usr/local/cuda/lib64/",
        ):
            try:
                return ctypes.CDLL(prefix + name)
            except OSError:
                continue
    raise SystemExit(
        "no NVTX library found (libnvtx3interop.so.1 / libnvToolsExt.so.1); "
        "install a CUDA toolkit or run --walls, which does not need NVTX"
    )


class _Nvtx:
    def __init__(self):
        lib = _load_nvtx()
        self._push = lib.nvtxRangePushA
        self._push.argtypes = [ctypes.c_char_p]
        self._pop = lib.nvtxRangePop

    def push(self, name: str):
        self._push(name.encode())

    def pop(self):
        self._pop()


class _Profiler:
    def __init__(self):
        cuda = ctypes.CDLL("libcuda.so.1")
        self.start = cuda.cuProfilerStart
        self.stop = cuda.cuProfilerStop


# ----------------------------------------------------------------------- capture


def _build_open(golden: str):
    """Run the prove up to the open and return a replayable open thunk.

    The sequence mirrors `prove_phase_bench.make_prove` (commit -> zerocheck
    -> lincheck -> open) — keep the two in sync; the bench is the authority on
    the phase bodies.
    """
    import frx

    frx.config.update("jax_enable_x64", True)
    import frx.numpy as fnp

    from flock_zorch import lincheck, prover, zerocheck
    from flock_zorch.challenger import Challenger
    from flock_zorch.pcs import ligerito as zorch_ligerito
    from flock_zorch.testing._util import await_all
    from flock_zorch.testing.prove_phase_bench import Circuit

    circ = Circuit("blake3")
    g = circ.ingest(golden)
    meta, cfg = g["meta"], g["cfg"]
    m, k_log, k_skip = meta["m"], meta["k_log"], meta["k_skip"]
    ir = k_log - k_skip
    circuit = circ.build(g)

    a_bits, b_bits, c_bits = (frx.device_put(x) for x in (g["a"], g["b"], g["z"]))
    # The bench uploads g["z"] twice (once as the zerocheck C track, once for
    # commit/open); here the open loop is the resident phase and the extra
    # 512 MiB at m32 matters, so reuse the C-track buffer — same host array.
    z = c_bits
    zlc = lincheck.stripe_to_device(g["zlc"], m, k_log)

    root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, z)
    ch = Challenger(circ.domain)
    prover.bind_statement(ch, g["stmt"], root)
    _, zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch)
    x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
    lc = lincheck.prove(zlc, None, None, x_ab, m, k_log, k_skip, ch=ch, circuit=circuit)
    await_all((zc, lc))
    # The open needs only z / pdata / the claim points; the zerocheck witness
    # tracks (1 GiB packed at m32) and the stripe would otherwise sit in the
    # BFC pool for the whole open loop and starve the open's first-call module
    # load of VRAM (the m32 OOM signature: "Failed to get module function").
    del a_bits, b_bits, zlc

    assert lc.claim is not None
    ab = fnp.concatenate([lc.claim.r_inner_rest, x_ab.x_outer], axis=0)
    cc = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
    t_snapshot = ch._t  # immutable pytree: assignment replays the transcript

    def open_once():
        ch._t = t_snapshot
        return await_all(prover.open_batch_ligerito(cfg, z, pdata, [ab, cc], ch))

    return open_once, meta


_SUBSTEPS = (
    ("flock_zorch.pcs.ring_switch", "prove_batched", "open/ring_switch"),
    ("flock_zorch.prover", "_combine_claims", "open/combine_claims"),
    ("flock_zorch.pcs.ligerito", "_open_jitted", "open/ligerito_jitted"),
    ("flock_zorch.pcs.ligerito", "_flock_proof_dict", "open/proof_dict"),
)


def _wrap_substeps(nvtx: _Nvtx):
    """Monkeypatch NVTX push/pop around the open's sub-steps.

    Wrapping at the callee modules (not `prover.open_batch_mixed_ligerito`'s
    locals) keeps the patch working regardless of import aliasing. The range
    closes at DISPATCH end — attribution is by device-timestamp containment,
    with the overlap caveats in the module docstring.
    """
    import importlib

    for mod_name, fn_name, label in _SUBSTEPS:
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)

        def wrapped(*args, __fn=fn, __label=label, **kwargs):
            nvtx.push(__label)
            try:
                return __fn(*args, **kwargs)
            finally:
                nvtx.pop()

        setattr(mod, fn_name, wrapped)


def run_capture(args) -> int:
    nvtx = _Nvtx()
    prof = _Profiler()
    open_once, meta = _build_open(args.golden)

    for _ in range(2):  # compile + warm
        open_once()

    _wrap_substeps(nvtx)
    prof.start()
    nvtx.push("open")
    open_once()
    nvtx.pop()
    prof.stop()
    print(f"captured one warm open iteration (m={meta['m']})")
    return 0


def run_walls(args) -> int:
    import numpy as np

    open_once, meta = _build_open(args.golden)
    for _ in range(2):
        open_once()

    ms = []
    for _ in range(args.walls):
        t0 = time.perf_counter()
        open_once()
        ms.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(ms)
    p10, med = np.percentile(arr, 10), np.median(arr)
    print(
        f"open m={meta['m']} walls over {args.walls} iters: "
        f"p10 {p10:.3f} ms | median {med:.3f} ms | min {arr.min():.3f} ms | "
        f"max {arr.max():.3f} ms | spread (max/min - 1) {arr.max() / arr.min() - 1:.1%}"
    )
    return 0


# ------------------------------------------------------------------------ report


def _kernels(db):
    """(host_enqueue_start, device_start, device_end, name) per kernel, ns.

    Attribution anchors on the HOST enqueue time (the CUDA runtime/driver API
    row joined via correlationId), not the device start: dispatch is async, so
    by the time a kernel RUNS the host has typically moved into a later NVTX
    range — device-timestamp containment silently re-attributes early
    sub-steps' kernels to later ones (measured here: it moved ring_switch's
    bit-select reduces into the jitted-open range). The enqueue anchor also
    fixes the pure_callback over-attribution the earlier method note warned
    about, since XLA thunk ranges wrap the enqueue as well.
    """
    api = dict(
        db.execute(
            "SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        ).fetchall()
    )
    return [
        (api.get(corr, ds), ds, de, name)
        for corr, ds, de, name in db.execute(
            """SELECT k.correlationId, k.start, k.end, s.value
               FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON k.demangledName = s.id"""
        )
    ]


def _nvtx_ranges(db):
    """(start, end, text) for push/pop ranges, both ours and XLA's thunks."""
    return db.execute(
        """SELECT start, end, COALESCE(text, s.value)
           FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
           WHERE eventType = 59 AND end IS NOT NULL"""
    ).fetchall()


def _base_kernel_name(name: str) -> str:
    """Collapse XLA's numbered fusion instances into one family."""
    import re

    return re.sub(r"_?\d+$", "", name.split("<")[0].strip())


def _hlo_op(text: str) -> str | None:
    import re

    m = re.search(r"hlo_op=([^#,]+)", text or "")
    return m.group(1) if m else None


def run_report(args) -> int:
    db = sqlite3.connect(args.report)
    kernels = _kernels(db)
    ranges = _nvtx_ranges(db)
    if not kernels:
        raise SystemExit(
            "no kernel rows — was the capture run with "
            "--cuda-graph-trace=node and --capture-range=cudaProfilerApi?"
        )

    ours = [r for r in ranges if r[2] and r[2].startswith("open")]
    thunks = [r for r in ranges if r[2] and "Thunk" in r[2]]

    def contained(host_t, rs):
        """Innermost (shortest) range containing a host enqueue timestamp."""
        best = None
        for s, e, t in rs:
            if s <= host_t < e and (best is None or e - s < best[1] - best[0]):
                best = (s, e, t)
        return best

    total_ms = sum(de - ds for _, ds, de, _ in kernels) / 1e6
    print(f"kernels {len(kernels)}, device busy {total_ms:.3f} ms")

    # -- per sub-step (host-enqueue containment against our NVTX ranges).
    per_step: defaultdict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for host_t, ds, de, _ in kernels:
        r = contained(host_t, ours)
        label = r[2] if r else "(outside open ranges)"
        per_step[label][0] += 1
        per_step[label][1] += (de - ds) / 1e6
    print("\nper sub-step:")
    for label, (n, ms) in sorted(per_step.items(), key=lambda x: -x[1][1]):
        print(
            f"  {label:<28} {ms:>9.3f} ms  {n:>5} kernels  {100 * ms / total_ms:.1f}%"
        )

    # -- top kernel families.
    fams: defaultdict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for _, ds, de, name in kernels:
        f = fams[_base_kernel_name(name)]
        f[0] += 1
        f[1] += (de - ds) / 1e6
    print("\ntop kernel families:")
    for name, (n, ms) in sorted(fams.items(), key=lambda x: -x[1][1])[: args.top]:
        print(f"  {name:<44} {ms:>9.3f} ms  n={n:<5} {ms * 1e3 / n:>8.1f} us/launch")

    # -- top HLO ops (XLA's own Thunk ranges; enqueue containment).
    hlo: defaultdict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for host_t, ds, de, _ in kernels:
        r = contained(host_t, thunks)
        op = _hlo_op(r[2]) if r else None
        if op:
            h = hlo[op]
            h[0] += 1
            h[1] += (de - ds) / 1e6
    if hlo:
        print("\ntop HLO ops (by enqueueing thunk):")
        for op, (n, ms) in sorted(hlo.items(), key=lambda x: -x[1][1])[: args.top]:
            print(f"  {op:<44} {ms:>9.3f} ms  n={n}")

    # -- per recursion level: the jitted open runs its levels sequentially and
    # each level ends in the host query-sampler callback (a pure_callback
    # thunk), so those thunk ranges inside the `open/ligerito_jitted` window
    # are the level fences.
    lig = next((r for r in ours if r[2] == "open/ligerito_jitted"), None)
    if lig is not None:
        fences = sorted(
            (s, e)
            for s, e, t in thunks
            if "pure_callback" in (t or "") and lig[0] <= s < lig[1]
        )
        lig_kernels = [
            k
            for k in kernels
            if (c := contained(k[0], ours)) and c[2] == "open/ligerito_jitted"
        ]
        if fences and lig_kernels:
            print(
                f"\nper level ({len(fences)} query-callback fences; "
                "kernels between consecutive fences = one level's enqueues):"
            )
            bounds = [lig[0]] + [f[0] for f in fences] + [lig[1]]
            for i in range(len(bounds) - 1):
                seg = [k for k in lig_kernels if bounds[i] <= k[0] < bounds[i + 1]]
                ms = sum(de - ds for _, ds, de, _ in seg) / 1e6
                label = f"level {i}" if i < len(fences) else "tail"
                print(f"  {label:<10} {ms:>9.3f} ms  {len(seg):>5} kernels")
                seg_fams: defaultdict[str, float] = defaultdict(float)
                for _, ds, de, name in seg:
                    seg_fams[_base_kernel_name(name)] += (de - ds) / 1e6
                for name, fms in sorted(seg_fams.items(), key=lambda x: -x[1])[:3]:
                    print(f"      {name:<40} {fms:>8.3f} ms")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--golden", default="blake3_ligerito_golden.bin")
    ap.add_argument("--walls", type=int, help="N un-profiled open iterations")
    ap.add_argument("--report", help="exported nsys sqlite to analyze")
    ap.add_argument("--top", type=int, default=12, help="rows in report tables")
    args = ap.parse_args()

    if args.report:
        return run_report(args)
    if args.walls:
        return run_walls(args)
    return run_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
