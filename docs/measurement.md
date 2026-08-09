# Measuring performance

Rules for producing a number this repo will believe, plus the environment facts
that block the tools outright. Each has cost a session at least once, and none
of them is discoverable by reading the code.

The benchmark itself — how to run it and what it has published — is in
[`README.md`](../README.md).

- **Verify `ptxas --version` says 13.3 before trusting any number.** With a 13.3
  `ptxas` on `PATH` the pinned frx wheel emits the hardware `clmad` GF(2¹²⁸)
  multiply; without it, the software `binary_field_ghash` multiply — same proof,
  no warning, and **5.5× on the whole prove**. `/usr/local/cuda` is not
  necessarily 13.3. The damage is non-uniform, so there is a cheap tell: `commit`
  degrades ~45× (it is almost pure F128 multiplies) while `zerocheck` moves only
  ~4×. **If `commit` is tens of ms at m28 instead of ~1.3 ms, that is a toolchain
  bug, not a perf finding.** Self-check by reproducing the README's published
  m28 baseline.
- **A `.bazelrc.user` `--override_module=zorch=...` silently substitutes the
  zorch you are measuring.** Before trusting any wall number, `git log` the
  override checkout against the `MODULE.bazel` pin. A stale override once hid a
  +35% m32 throughput difference (#200 erratum) — every m32 wall measured under
  it had to be thrown away.
- **Do not set `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` by default.** At m32 it
  *inflates* the prove **~14%** — 71.8 ms without it vs 81.6 ms with it, means
  of three fresh processes per arm, `--throughput` best-of-10 each,
  idle RTX 5090 — and it is also what makes the barriered phase-split mode OOM
  in its warm-up prove. `XLA_PYTHON_CLIENT_PREALLOCATE=false` alone suffices at
  m32. Reach for the async allocator only against an actual allocator OOM. The
  size of the penalty moves with the phase mix (#200 measured +21–31% on an
  earlier pin), so re-price it rather than quoting a fixed number.
- **Pick targets by marginal µs/hash across a size step, never by phase share at
  one `m`.** Phase shares at small `m` are dominated by fixed floors and give the
  wrong ranking for a throughput goal: `open` has read as high as 61–67% at
  m ≤ 28 and roughly half that at m32, because its launch-latency floor
  amortizes. Compute
  `Δphase_ms / Δhashes` across two sizes; a term whose per-hash cost *rises* sets
  the ceiling, one that *falls* is a floor being spread. The marginal predicted
  the m32 ranking two size steps early — the absolute share never did.
- **A phase is not a target.** `zerocheck` bundles the round-1 URM extend and the
  multilinear ladder, whose relative sizes invert with `m`. Split to the prover
  round before scoping work off a phase number.
- **Latency vs arithmetic: busy from nsys, wall from a clean run.** nsys inflates
  *host* dispatch ~2×, so inter-kernel gaps on its timeline are contaminated;
  on-device kernel durations are not. Pass `--cuda-graph-trace=node` or
  CUDA-graph dispatches under-report kernels ~50×. Busy reproduces to <0.1%
  while a phase wall spreads several percent, so read busy for small deltas.
  Use `--throughput` for any number compared against a goal — the barriered
  phase-split mode runs ~14% slower and is for attribution only.
- **A/B knobs in combination, not one at a time.** Tiling the elements reduce
  measured flat and a mask-select measured *slower*, which read as "not
  tunable" — then zorch#590 combined both with a per-program parity fold for
  −18%. Each measurement was right and the conclusion was wrong. One knob at a
  time can only refute one knob.
- **A profiler's stall reason says where warps wait, not what is fixable.** Read
  Speed-of-Light first: it named instruction count on a kernel whose loudest
  stall line pointed at loads. And profile the slow kernel's *neighbours in the
  same capture* — siblings at 76–86% DRAM turn "is 707 GB/s good?" from a
  judgement call into a measurement.
- **ncu needs `sudo` on the build box** (`ERR_NVGPUCTRPERM`, all-or-nothing —
  it refuses even `--section LaunchStats`). nsys is unaffected because it uses
  CUPTI tracing rather than hardware counters. Under sudo, pin `HOME` and use
  absolute paths, or the env reset sends `CUDA_ROOT` and the venv to root's
  home and you measure the software-GHASH path.
