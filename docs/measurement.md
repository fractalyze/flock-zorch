# Measuring performance

Rules for producing a number this repo will believe, plus the environment facts
that block the tools outright. Each has cost a session at least once, and none
of them is discoverable by reading the code.

The benchmark itself — how to run it and what it has published — is in
[`README.md`](../README.md).

- **The `ptxas` that decides `clmad` is `CUDA_ROOT`'s, not `PATH`'s — so
  `ptxas --version` is not the check.** With a 13.3 `ptxas` the compiler emits
  the hardware `clmad` GF(2¹²⁸) multiply; without it, the software
  `binary_field_ghash` multiply — same proof, no warning, and **5.5× on the
  whole prove at m28, more as `m` grows** (measured ~16× at m32). The damage is
  non-uniform, so there is a cheap tell: `commit` degrades ~45× (it is almost
  pure F128 multiplies) while `zerocheck` moves only ~4×. **If `commit` is tens
  of ms at m28 instead of ~1.3 ms, that is a toolchain bug, not a perf finding.**
  Self-check by reproducing the README's published m28 baseline.

  frx sets XLA's `xla_gpu_cuda_data_dir` from `CUDA_ROOT` itself, and XLA
  prefers `<that dir>/bin/ptxas` over anything on `PATH`. Three consequences,
  each of which has burned a session:
  - Unset, it falls back to the venv's bundled CUDA (`nvidia/cuda_nvcc`,
    currently 12.9) — so `ptxas --version` can say 13.3 while XLA compiles 12.9.
    `/usr/local/cuda` is not necessarily 13.3 either.
  - `--xla_gpu_cuda_data_dir` in `XLA_FLAGS` does **not** work: the flag parses,
    then frx overwrites the field. And `CUDA_ROOT` is read when frx is imported,
    so setting it from inside Python after `import frx` is a no-op — export it
    before the process starts.
  - `xla_gpu_cuda_data_dir` is deliberately excluded from the persistent
    compilation-cache key, so after fixing `CUDA_ROOT` a cached executable built
    on the software path is still a hit. Clear the cache (or run without one)
    before re-measuring.

  `TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_MAX_VLOG_LEVEL=1` dumps every debug option by
  name: grep the log for `xla_gpu_cuda_data_dir` (must be your 13.3 tree) and
  `Targeting PTX version: 93`. Both gates read that one ptxas probe, so a 9.3
  header proves the toolchain is wired right. `XLA_FLAGS=--xla_dump_to=<dir>`
  then confirms the kernel itself contains `clmad.{lo,hi}.u64` — absent there
  means you dumped a kernel with no GHASH multiplies, not a toolchain problem.
  (A `libdevice not found` warning is separate and only fatal if a fusion needs
  one; a 13.3 tree without `nvvm/libdevice` wants one merged in from a 12.x.)

  All of this is identical for the shipped wheel and a self-built plugin — the
  path lives in frx's Python layer, not the `.so`. A plugin built against
  hermetic CUDA 12.9 is **not** capped, because the PTX header is raised off the
  runtime ptxas alone; rebuilding it against 13.3 is not the fix.
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

  Three rules follow from that, each of which has now cost a session:
  - **Compare busy and wall within the SAME iteration.** A profiled capture's
    busy against a clean `--walls` number mixes two execution modes and can
    manufacture an idle bucket that is not there. Gate the capture first: the
    profiled iteration's own wall must land near the clean wall before anything
    derived from it is trusted.
  - **Profile ≥2 iterations and discard the first.** The first pays a one-time
    driver cost that is indistinguishable from a workload stall — m32 `open`
    measured 85.0 / 23.3 / 21.7 ms across three, the first carrying a single
    47 ms contiguous device-idle gap during which the host issued *no* CUDA
    calls. `nsys_capture.py --window` profiles exactly one, so drive its
    internals (`WINDOWS`, `_Nvtx`, `_Profiler`, `_wrap_substeps`) for a
    multi-iteration capture.
  - **A profiler cannot answer launch-structure questions at all.** XLA runs a
    command buffer as a plain thunk sequence whenever a profiler session is
    active (`xla_enable_command_buffers_during_profiling` defaults false —
    upstream TODO b/290773547), so every capture of this prover executes
    kernel-by-kernel and any "which op splits the command buffer" or
    per-boundary cost describes the fallback, not production. Setting the flag
    does not rescue it: `--cuda-graph-trace=node` instruments each node and
    gives the graph's advantage back (m32 `open` 25.1 / 22.5 ms with the flag
    vs 23.3 / 21.7 without). Use an un-profiled A/B instead —
    `XLA_FLAGS=--xla_gpu_enable_command_buffer=` against the default. On the
    m32 `open` window that is p10 19.647 (graphs on) vs 19.727 ms (off), i.e.
    CUDA graphs are worth ~0 there, which refuted a launch-collapse plan before
    any code was written.
- **A/B knobs in combination, not one at a time.** Tiling the elements reduce
  measured flat and a mask-select measured *slower*, which read as "not
  tunable" — then zorch#590 combined both with a per-program parity fold for
  −18%. Each measurement was right and the conclusion was wrong. One knob at a
  time can only refute one knob.
- **Never derive DRAM traffic from HLO shapes. `ERR_NVGPUCTRPERM` does not mean
  ncu is blocked — it means ncu needs one `sudo` (see the ncu bullet below), so
  ask for it.**

  This matters because the substitute is worse than it looks, and reaching for
  it has cost **three** sessions. A one-read-one-write shape model under-counts
  whenever a kernel re-reads, over-fetches, or writes a buffer its root op does
  not imply — and the error direction is systematic: fewer bytes over the same
  wall reads as *low efficiency*, which manufactures headroom that is not there.
  #213 priced a reduce at 56% of peak from shapes, corrected itself to 47%, and
  was wrong both times; ncu measured **95%**, with the kernel moving 2.00× the
  required bytes. Reading that as a redundant pass rather than an efficiency gap
  is what produced the actual fix (#215). The commit-phase work then repeated it
  twice in one table — a per-launch byte count charged against two launches'
  aggregate time (40% claimed, 76–86% measured), and a fusion emitting a tuple
  output the model did not know about (54% claimed, 81.9% measured) — after
  recording "ncu is blocked on this box" as the reason. It was not blocked.

  When you genuinely cannot measure (someone else's box, a shipped wheel), the
  derived number is a **lower bound on efficiency / upper bound on headroom**,
  never evidence that a kernel is slow. Three checks, in order: count bytes
  **per launch** and multiply by the launch count nsys reports; open the fused
  computation and check whether it emits more buffers than its name implies; and
  when one kernel reads as far off peak while its same-shape neighbours do not,
  suspect the byte count before the kernel.
- **A profiler's stall reason says where warps wait, not what is fixable.** Read
  Speed-of-Light first: it named instruction count on a kernel whose loudest
  stall line pointed at loads. And profile the slow kernel's *neighbours in the
  same capture* — siblings at 76–86% DRAM turn "is 707 GB/s good?" from a
  judgement call into a measurement.
- **ncu needs `sudo` on the build box** (`ERR_NVGPUCTRPERM`, all-or-nothing —
  it refuses even `--section LaunchStats`). Confirm it in one line rather than
  inferring it from a failed run: `grep RmProfilingAdminOnly
  /proc/driver/nvidia/params`, where `1` means counters are admin-only. nsys is
  unaffected because it uses CUPTI tracing rather than hardware counters.
  Under sudo, pin `HOME` and use absolute paths, or the env reset sends
  `CUDA_ROOT` and the venv to root's home and you measure the software-GHASH
  path. Two more, once it runs: `--profile-from-start no` honours
  `nsys_capture.py`'s `cuProfilerStart/Stop`, so warm-up and autotune are
  excluded by construction; and `ncu --csv` emits a **units row after the
  header**, so parsing the header alone reads every metric as zero.
