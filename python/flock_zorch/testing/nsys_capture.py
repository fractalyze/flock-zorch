# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""nsys capture + attribution harness for one prover window, and a wall sampler.

`prove_phase_bench` answers which PHASE owns the time. This answers the next
question down — which kernels inside one window own it — which
`docs/measurement.md` insists on before work is scoped ("a phase is not a
target"). The capture scaffolding for this had been rebuilt from session
scratch several times over, in more than one perf lane, which is why it lives
in the tree.

Windows (`--window`, default `open`):

- **`open`** — the Ligerito recursive open, with sub-step ranges around
  `ring_switch.prove_batched`, `_combine_claims`, `_open_jitted` and
  `_flock_proof_dict`, plus a per-recursion-level breakdown. Run it with
  `--fs blake3` to attribute the deficit that arm carries: the barriered phase
  split puts ~85% of it in THIS window (m32, post-#271).
- **`commit`** — the L0 commit, with no sub-step ranges: it is one jitted
  program, so its encode / layout-assembly / leaf-hash groups have to be read
  off the kernel names in the report rather than fenced by NVTX. `--tree`
  selects the Merkle arm, and a commit number is only comparable to another on
  the SAME arm: the two differ in kind, not by a constant.
- **`zerocheck-urm`**, **`zerocheck-ml`** — one zerocheck `ProverRound` each.
  Zerocheck is why a window is round-level and not phase-level: its two rounds
  turned out to have OPPOSITE bindings — the round-1 URM composite
  arithmetic-bound at ~9x its read roofline, the multilinear ladder
  bandwidth-bound with its folds already at 90% of peak — so a zerocheck-level
  number cannot pick either one's lever.

Modes:

- **capture** (default): run the prove up to the window, warm it, then replay
  ONE iteration inside a `cudaProfilerApi` range with NVTX sub-step ranges.
  Run it UNDER nsys — recipe below. Gating the range excludes warm-up and
  autotune by CONSTRUCTION rather than filtering them out afterwards, which is
  what makes the bucket table trustworthy.
- **--walls N**: the same window replayed N times un-profiled, reporting
  p10 / median / min. This is the clean-wall channel; nsys inflates host
  dispatch ~2x, so wall numbers must never come from the capture run.
- **--report FILE.sqlite**: read an exported capture and print device-busy
  attribution: per sub-step, per CUDA kernel, per XLA HLO-op family, and (for
  `open`) per recursion level.

Every window is replayable because either challenger advances by REPLACING its
immutable `_t` pytree: snapshotting `_t` and re-assigning it before each
iteration replays byte-identical Fiat-Shamir draws, so every iteration runs the
same program on the same data. Rebuilding the state per iteration instead would
pay a fresh commit each time.

Attribution method: `busy` = sum of on-device kernel durations, `wall` = clean
un-profiled runs, `latency` = wall - busy. Kernels are attributed to NVTX
ranges by HOST-ENQUEUE containment — each kernel's CUDA API row (joined via
correlationId) locates the dispatch inside the range that issued it. This
deliberately differs from the device-timestamp containment earlier sessions
used: dispatch is async, so device-time containment re-attributes an early
sub-step's kernels to whichever range the host had reached by the time they ran
(measured: it moved ring_switch's bit-select reduces into the jitted-open
range), and it lands kernels that overlap a blocking `pure_callback` inside the
callback's thunk range. The enqueue anchor fixes both.

`--cuda-graph-trace=node` is mandatory in the capture: without it XLA's
CUDA-graph dispatches under-report kernels ~50x.

Where this lives, and when to move it. Everything above the window registry is
flock-agnostic — it knows CUPTI, NVTX, sqlite and XLA thunk ranges, nothing
about this prover. Only the registry entries do, and they reach into private
internals to do it. So the file is deliberately split along that line, and the
split is the whole cost of promoting it later.

It stays here because the demand is here: at the time of writing the sibling
frx consumers (hash-frx, sig-frx, enc-frx) carry no profiling code at all, and
the accumulated nsys method notes came almost entirely from this repo. **When a
second repo needs this, promote the machinery** — `zkbench-py` is the natural
home, since it is already a shared benchmarking package and lands the tool
without adding Fractalyze-only surface to the frx fork or forcing a dependency
on zorch's SNARK spine for what is only a profiling helper. Leave the window
registry behind; it is flock's by nature.

Run (the toolchain preamble is load-bearing — ptxas 12.9 silently selects
soft-GHASH, 5.5x on the prove; `commit` at tens of ms instead of ~1.3 ms at
m28 is the tell. See docs/measurement.md, which also says NOT to set
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async):

    export CUDA_ROOT=$HOME/.local/cuda13   # must be a 13.3 ptxas
    export FRX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
    unset JAX_PLATFORMS XLA_PYTHON_CLIENT_ALLOCATOR JAX_COMPILATION_CACHE_DIR
    export PATH="$CUDA_ROOT/bin:$PATH"
    PY="python:$(scripts/zorch_pythonpath.sh)"
    H=python/flock_zorch/testing/nsys_capture.py

    # clean walls (no nsys):
    PYTHONPATH="$PY" .venv/bin/python "$H" --window zerocheck-ml \
        --golden blake3_ligerito_golden_m32.bin --walls 12

    # capture:
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        --cuda-graph-trace=node --trace=cuda,nvtx -o zcml_m32 \
        --force-overwrite true env PYTHONPATH="$PY" .venv/bin/python "$H" \
        --window zerocheck-ml --golden blake3_ligerito_golden_m32.bin
    nsys export --type sqlite --force-overwrite true -o zcml_m32.sqlite \
        zcml_m32.nsys-rep

    # report (pass the same --window so sub-step ranges are picked up):
    .venv/bin/python "$H" --window zerocheck-ml --report zcml_m32.sqlite
"""
from __future__ import annotations

import argparse
import ctypes
import sqlite3
import time
from collections import defaultdict
from typing import Any, NamedTuple

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


class Arms(NamedTuple):
    """Which Merkle and Fiat-Shamir hashes a window runs.

    Two axes, selected independently, because they answer different questions
    and differ by two orders of magnitude: the Merkle arm's whole cost is a few
    ms of commit device-busy, while the FS arm drives nearly all of the BLAKE3
    deficit, in `open`. Collapsing them into one flag would leave the small term
    inseparable from the large one — which is how a Merkle change once got
    credited with a figure it could not have produced. `--fs blake3 --tree
    blake3` is the profile the 5M benchmark is scored on.
    """

    tree: str
    fs: str


def _resolve_tree(name: str):
    """Merkle arm by name -> the tree object the ligerito seams take.

    Resolved inside a builder rather than at import: `hash.merkle` pulls in frx,
    which every builder configures for x64 first. Both arms hash by type, so
    naming one explicitly is the same jit cache key as letting the seam default.
    """
    from flock_zorch.hash import merkle

    return {
        "sha256": merkle.GHASH_SHA256_TREE,
        "blake3": merkle.GHASH_BLAKE3_TREE,
    }[name]


def _resolve_fs(name: str):
    """Fiat-Shamir arm by name -> the challenger class.

    Both classes advance by REPLACING an immutable `_t`, which is what the
    windows' replay contract needs — so either arm is snapshot-replayable on
    the same terms.
    """
    from flock_zorch.blake3_challenger import Blake3DeviceChallenger
    from flock_zorch.sha256_challenger import Sha256Challenger

    return {"sha256": Sha256Challenger, "blake3": Blake3DeviceChallenger}[name]


def _build_open(golden: str, arms: Arms):
    """Run the prove up to the open and return a replayable open thunk.

    The sequence mirrors `prove_phase_bench.make_prove` (commit -> zerocheck
    -> lincheck -> open) — keep the two in sync; the bench is the authority on
    the phase bodies.
    """
    import frx

    frx.config.update("jax_enable_x64", True)
    import frx.numpy as fnp

    from flock_zorch import lincheck, prover, zerocheck
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

    arm = _resolve_tree(arms.tree)
    root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, z, arm)
    ch = _resolve_fs(arms.fs)(circ.domain)
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
        return await_all(prover.open_batch_ligerito(cfg, z, pdata, [ab, cc], ch, arm))

    return open_once, meta


def _build_commit(golden: str, arms: Arms):
    """Run only the ingest, then return a replayable thunk for the `commit`
    phase — `commit_flock_ligerito`, the same call `prove_phase_bench` times.

    No sub-steps, deliberately. `zorch.pcs.ligerito.prover._commit` is one
    `frx.jit` program, so encode / layout-assembly / leaf compression are
    fused into a single dispatch and no Python-level NVTX range can separate
    them. The decomposition has to come from grouping the window's kernels by
    name in the report, which is how #205 produced its 40.2 / 50.9 / 7.8 split
    and is what makes this window's numbers comparable to it.

    `--tree` picks the arm, and the two are not interchangeable as targets: the
    SHA-256 arm is flock's default, but the 5M BLAKE3/s benchmark commits with
    BLAKE3 (`prover.BLAKE3_PROFILE`), whose leaf level is one fused
    `hash_frx.blake3` composite rather than a pad/pack assembly. A commit number
    is only comparable to another taken on the same arm.

    commit is the one phase with no prior state to rebuild: it consumes the
    witness straight off the golden, so unlike the open/zerocheck windows this
    needs no transcript snapshot to replay.
    """
    import frx

    frx.config.update("jax_enable_x64", True)

    from flock_zorch.pcs import ligerito as zorch_ligerito
    from flock_zorch.testing._util import await_all
    from flock_zorch.testing.prove_phase_bench import Circuit

    circ = Circuit("blake3")
    g = circ.ingest(golden)
    meta, cfg = g["meta"], g["cfg"]
    z = frx.device_put(g["z"])

    arm = _resolve_tree(arms.tree)

    def commit_once():
        return await_all(zorch_ligerito.commit_flock_ligerito(cfg, z, arm))

    return commit_once, meta


def _build_zerocheck(golden: str, arms: Arms, which: str, equal_factors: bool = False):
    """Run the prove up to one zerocheck `ProverRound` and return a replayable
    thunk for that round — `urm` for round-1's univariate skip, `ml` for the
    multilinear ladder.

    `equal_factors=True` aliases the B track to the SAME device array as A,
    which is what routes the multilinear tail onto the equal-factor
    squared-sum ladder (`b_rows is a_rows`). The proved instance is then
    a·a ⊕ c — NOT the golden's proof, so this mode is a wall/attribution
    channel only (it sizes the sq ladder on a real witness at the golden's m);
    the identity e2e byte gate pins the sq wire separately.

    `zerocheck` is the case that forces a round-level window rather than a
    phase-level one: its two rounds turned out to have opposite bindings — the
    round-1 URM composite arithmetic-bound at ~9x its read roofline, the ladder
    bandwidth-bound with its folds already at 90% of peak — so a zerocheck-level
    number cannot pick either one's lever, and a diagnosis carried from one
    round to the other is simply wrong.

    `prove_rounds` is a bare loop over `rnd(carry, transcript)`, so driving the
    rounds one at a time here is the shipped path rather than an approximation
    of it. The carry construction mirrors `zerocheck.prove_packed` and has to
    follow it if that changes shape.
    """
    import frx

    frx.config.update("jax_enable_x64", True)
    import frx.numpy as fnp

    from flock_zorch import lincheck, prover
    from flock_zorch.pcs import ligerito as zorch_ligerito
    from flock_zorch.testing._util import await_all
    from flock_zorch.testing.prove_phase_bench import Circuit
    from flock_zorch.zerocheck import prover as zcp

    circ = Circuit("blake3")
    g = circ.ingest(golden)
    meta, cfg = g["meta"], g["cfg"]
    m, k_log = meta["m"], meta["k_log"]

    a_bits, b_bits, c_bits = (frx.device_put(x) for x in (g["a"], g["b"], g["z"]))
    if equal_factors:
        b_bits = a_bits  # alias, don't copy — `is` routes the sq ladder
    z = c_bits  # same host array as the C track; saves 512 MiB at m32
    lincheck.stripe_to_device(g["zlc"], m, k_log)

    root, _pdata = zorch_ligerito.commit_flock_ligerito(
        cfg, z, _resolve_tree(arms.tree)
    )
    ch = _resolve_fs(arms.fs)(circ.domain)
    prover.bind_statement(ch, g["stmt"], root)

    k_skip = zcp.K_SKIP
    r_skip, r_outer = zcp.sample_challenge_coords(ch, m, k_skip)
    r = fnp.concatenate([r_skip, zcp._SMALL_G, zcp._MEDIUM_G, r_outer])
    carry = zcp._ZerocheckCarry(a_bits, b_bits, c_bits, r=r)
    urm, ml = zcp.zerocheck_steps(m, k_skip)
    if which == "ml":
        carry, ch, _ = urm(carry, ch)
        await_all(carry)  # the URM is settled OUTSIDE the measured window
    rnd = ml if which == "ml" else urm

    # Same replay trick as the open: the rounds are functional in `carry`, and
    # the challenger advances by REPLACING its immutable `_t`, so restoring the
    # snapshot replays byte-identical Fiat-Shamir draws. Rebuilding the whole
    # state per iteration instead would pay a fresh commit each time.
    t_snapshot = ch._t

    def round_once():
        ch._t = t_snapshot
        return await_all(rnd(carry, ch))

    return round_once, meta


_SUBSTEPS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "open": (
        ("flock_zorch.pcs.ring_switch", "prove_batched", "open/ring_switch"),
        ("flock_zorch.prover", "_combine_claims", "open/combine_claims"),
        ("flock_zorch.pcs.ligerito", "_open_jitted", "open/ligerito_jitted"),
        ("flock_zorch.pcs.ligerito", "_flock_proof_dict", "open/proof_dict"),
    ),
    # Patch on `zerocheck.prover`, not on `._fold`: the round body looks these
    # up in its own module globals, so that is where the wrapper must land.
    "zerocheck-ml": (
        ("flock_zorch.zerocheck.prover", "_lagrange_weights", "zerocheck-ml/weights"),
        ("flock_zorch.zerocheck.prover", "_fold_at_z", "zerocheck-ml/fold_at_z"),
        ("flock_zorch.zerocheck.prover", "_EQ_TABLES", "zerocheck-ml/eq_tables"),
        ("flock_zorch.zerocheck.prover", "_mlv_sumcheck", "zerocheck-ml/mlv_sumcheck"),
    ),
    "zerocheck-ml-sq": (
        (
            "flock_zorch.zerocheck.prover",
            "_lagrange_weights",
            "zerocheck-ml-sq/weights",
        ),
        ("flock_zorch.zerocheck.prover", "_fold_at_z", "zerocheck-ml-sq/fold_at_z"),
        ("flock_zorch.zerocheck.prover", "_SQRT", "zerocheck-ml-sq/sqrt"),
        (
            "flock_zorch.zerocheck.prover",
            "_EQ_TABLES_SQ",
            "zerocheck-ml-sq/eq_tables",
        ),
        (
            "flock_zorch.zerocheck.prover",
            "_mlv_sumcheck_sq",
            "zerocheck-ml-sq/mlv_sumcheck_sq",
        ),
    ),
    "zerocheck-urm": (
        ("flock_zorch.zerocheck._urm", "round1_rows", "zerocheck-urm/round1_rows"),
        (
            "flock_zorch.zerocheck.prover",
            "_interpolate_at_z_on_lambda",
            "zerocheck-urm/interpolate",
        ),
    ),
}

# name -> (build thunk, sub-steps). The capture/walls/report machinery below is
# window-agnostic; only these two entries differ per window.
WINDOWS: dict[str, tuple[Any, tuple[tuple[str, str, str], ...]]] = {
    "open": (_build_open, _SUBSTEPS["open"]),
    "commit": (_build_commit, ()),
    "zerocheck-urm": (
        lambda golden, arms: _build_zerocheck(golden, arms, "urm"),
        _SUBSTEPS["zerocheck-urm"],
    ),
    "zerocheck-ml": (
        lambda golden, arms: _build_zerocheck(golden, arms, "ml"),
        _SUBSTEPS["zerocheck-ml"],
    ),
    "zerocheck-ml-sq": (
        lambda golden, arms: _build_zerocheck(golden, arms, "ml", equal_factors=True),
        _SUBSTEPS["zerocheck-ml-sq"],
    ),
}


def _wrap_substeps(nvtx: _Nvtx, substeps):
    """Monkeypatch NVTX push/pop around the open's sub-steps.

    Wrapping at the callee modules (not `prover.open_batch_mixed_ligerito`'s
    locals) keeps the patch working regardless of import aliasing. The range
    closes at DISPATCH end — attribution is by device-timestamp containment,
    with the overlap caveats in the module docstring.
    """
    import importlib

    for mod_name, fn_name, label in substeps:
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
    build, substeps = WINDOWS[args.window]
    nvtx = _Nvtx()
    prof = _Profiler()
    once, meta = build(args.golden, Arms(tree=args.tree, fs=args.fs))

    for _ in range(2):  # compile + warm
        once()

    _wrap_substeps(nvtx, substeps)
    prof.start()
    nvtx.push(args.window)
    once()
    nvtx.pop()
    prof.stop()
    print(f"captured one warm {args.window} iteration (m={meta['m']})")
    return 0


def run_walls(args) -> int:
    import numpy as np

    build, _ = WINDOWS[args.window]
    once, meta = build(args.golden, Arms(tree=args.tree, fs=args.fs))
    for _ in range(2):
        once()

    ms = []
    for _ in range(args.walls):
        t0 = time.perf_counter()
        once()
        ms.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(ms)
    p10, med = np.percentile(arr, 10), np.median(arr)
    print(
        f"{args.window} m={meta['m']} walls over {args.walls} iters: "
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

    ours = [r for r in ranges if r[2] and r[2].startswith(args.window)]
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
    if lig is not None:  # open-only: no other window has recursion levels
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
    ap.add_argument(
        "--window",
        choices=tuple(WINDOWS),
        default="open",
        help="which prover window to isolate",
    )
    ap.add_argument(
        "--tree",
        choices=("sha256", "blake3"),
        default="sha256",
        help="Merkle arm to commit with; sha256 is flock's default, blake3 is "
        "what the 5M BLAKE3/s benchmark profile commits with",
    )
    ap.add_argument(
        "--fs",
        choices=("sha256", "blake3"),
        default="sha256",
        help="Fiat-Shamir arm. Independent of --tree so each can be priced "
        "alone; pass both as blake3 for the profile the benchmark is scored on",
    )
    ap.add_argument("--walls", type=int, help="N un-profiled iterations")
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
