# Measuring performance

Rules for producing a number this repo will believe, plus the environment facts
that block the tools outright. Each has cost a session at least once, and none
of them is discoverable by reading the code.

The benchmark itself — how to run it and what it has published — is in
[`README.md`](../README.md).

- **Put ONE 13.3+ toolchain first on `PATH` and set nothing else.** `ptxas` and
  `nvlink` are resolved from *different* places, so a half-set environment
  mixes toolchains: `ptxas` follows `xla_gpu_cuda_data_dir` (which frx sets
  from an exported `CUDA_ROOT`), while the linker ignores that directory and
  takes `PATH`, then `/usr/local/cuda`. Exporting only `CUDA_ROOT` therefore
  assembles PTX 9.3 with 13.3 and links it with 12.9 — which does not degrade
  quietly: the m30 Ligerito gate dies in `nvlink fatal: Internal FNLZR error`,
  a failure that reads as a byte regression in whatever merged last (this cost
  a filed issue, fractalyze/flock-zorch#272, and a session). One toolchain on
  `PATH` satisfies both lookups, and the `*_oracle_test.py` gates now refuse
  the mixed state up front. Beware
  `export CUDA_ROOT=<root> PATH="$CUDA_ROOT/bin:$PATH"` as a single statement:
  `$CUDA_ROOT` there still expands to its OLD value, so the root never reaches
  `PATH` — that one line is how the mixed state is usually reached.

- **A 13.3 `ptxas` is what emits `clmad`, and `ptxas --version` is not
  automatically the check** — it is only the check once the toolchain is on
  `PATH` as above. With a 13.3 `ptxas` the compiler emits
  the hardware `clmad` GF(2¹²⁸) multiply; without it, the software
  `binary_field_ghash` multiply — same proof, no warning, and **5.5× on the
  whole prove at m28, more as `m` grows** (measured ~16× at m32). The damage is
  non-uniform, so there is a cheap tell: `commit` degrades ~45× (it is almost
  pure F128 multiplies) while `zerocheck` moves only ~4×. **If `commit` is tens
  of ms at m28 instead of ~1.3 ms, that is a toolchain bug, not a perf finding.**
  Self-check by reproducing the README's published m28 baseline.

  frx sets XLA's `xla_gpu_cuda_data_dir` from `CUDA_ROOT` itself, and XLA
  prefers `<that dir>/bin/ptxas` over anything on `PATH` — for **ptxas only**.
  Four consequences, each of which has burned a session:
  - With `CUDA_ROOT` unset, ptxas comes off `PATH`, falling back to the venv's
    bundled CUDA (`nvidia/cuda_nvcc`, currently 12.9) when `PATH` has none — so
    `ptxas --version` can say 13.3 while XLA compiles 12.9 if the 13.3 tree is
    not the one `PATH` resolves. `/usr/local/cuda` is not necessarily 13.3.
  - The linker does NOT follow `xla_gpu_cuda_data_dir`. `nvlink` comes from
    `PATH`, then `/usr/local/cuda` — so `CUDA_ROOT` alone cannot supply it, and
    a 13.3 `CUDA_ROOT` with a bare `PATH` links 13.3 cubins with 12.9. Measured
    by adding *only* a 13.3 `nvlink` to `PATH` in that state: the m30 gate goes
    from `nvlink fatal` to PASS with nothing else changed.
  - `--xla_gpu_cuda_data_dir` in `XLA_FLAGS` does **not** work: the flag parses,
    then frx overwrites the field. And `CUDA_ROOT` is read when frx is imported,
    so setting it from inside Python after `import frx` is a no-op — export it
    before the process starts.
  - `xla_gpu_cuda_data_dir` is deliberately excluded from the persistent
    compilation-cache key, so after fixing `CUDA_ROOT` a cached executable built
    on the software path is still a hit. Clear the cache (or run without one)
    before re-measuring.

  `TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_MAX_VLOG_LEVEL=1` dumps every debug option by
  name: grep the log for `Targeting PTX version: 93`, which is the direct
  reading of what ptxas emitted. Do NOT require `xla_gpu_cuda_data_dir` to name
  your 13.3 tree — on the recommended `PATH`-only setup it stays at its default
  `./cuda_sdk_lib` while PTX 93 is emitted anyway. And a 9.3 header proves only
  the *assembler*; `nvlink --version` is a separate check.
  `XLA_FLAGS=--xla_dump_to=<dir>`
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
- **Every A/B needs a quantity the change cannot touch, measured in the same
  window.** Order-alternation, min-of-N and in-process interleaving all reduce
  drift; none of them tell you whether what is left is signal — only a control
  does. For a change confined to one phase the control is free, because
  `prove_phase_bench` prints the others: ablate one sub-step of `open` and
  `zerocheck` must not move. Reject the run if it moves more than ~5% rather than
  averaging it in. Twice in one session this was the only thing that caught a
  fiction: an in-process unroll sweep read 1.558× while its bandwidth-bound
  control kernel swung 28%, and an `open` ablation read *higher with work
  removed* while `zerocheck` — untouchable by it — moved 44%.
- **Size a component's share from a speedup you already have, before optimizing
  it again.** When both a component's speedup and the whole-system speedup are
  recorded, Amdahl gives the share for free: prime-ir#432's 25.9× on the GF
  multiply bought 3.55× on the prove, so the multiply was 74.7% before and
  ~10% after — i.e. **making it free now buys ~12%**. Three later attempts at
  the multiply (32-bit limbs, bit-slicing, nibble tables) were capped at that
  before they started.
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
- **A microbenchmark may motivate a hypothesis; it must never size one.** The
  same marked transcript region prices at ~44–150 µs in a tight chain and
  ~1.0–2.1 ms in situ, 30–40× apart, because the chain pipelines with the state
  hot while the prover's sits between large kernels. Three explanations for that
  gap were each built on a clean, well-controlled bench and each refuted by the
  real prove — the last one, "the per-hop `frx.jit` boundary costs ~1 ms",
  matched the prover's figure to a digit and was still wrong (unwrapping all
  eleven wrappers is *slower*). Size a lever by ablating the real prover — the
  ablation harness's gates cost one run each — and treat any bench number that
  arrives already matching your expectation as the one most in need of a check.
  A control that is the identity element of the operation under test is not a
  control: `acc * one` is folded away, and it inflated a "serialization cost"
  by 10×.
- **Comparing two backends means comparing shares, not absolutes.** Metal
  `open` at m28 against CUDA `open` is an M4 Pro laptop GPU against an RTX 5090
  — the 3.5× between them is hardware and supports nothing. The same change
  expressed as a share of its own phase (17.5% on Metal, 1.8% on CUDA) cancels
  the hardware and is the comparison that carries information.
- **Pair the two arms inside ONE process.** A cross-process A/B recompiles the
  whole prove per arm (~5 min here), which stretches the pair's window until the
  host drifts inside it — 0 of 6 pairs survived; the in-process version of the
  same comparison got 12 of 19. And the control should be **one large phase the
  change cannot reach**, not a max over several small ones: `zerocheck` alone
  keeps ~9 of 12 pairs where a max over `commit`/`zerocheck`/`lincheck` rejected
  9 of 10 on the small phases' own noise.
- **A worktree at the right sha is not the right stack — prove the emitter
  claimed the marker.** Source pins and wheel pins drift independently: the
  ablation harness reads the source, the GPU reads the plugin. A stage-2 lever
  here was sized at +13.00 ms (m=28) → +46.10 ms (m=31) and approved for an
  emitter on that rise; the measuring venv was eight days stale and its
  recognizer did not know the round's `variant`, so the round ran **de-fused in
  both arms**. Re-measured where it fuses, the same arm gives +6.66 → +7.86 ms
  — a 1.18× rise, not 3.55×, and the decision inverts. Rejecting it as "applied
  to both arms, so it cancels" is only valid for an arm asymmetry; when the
  disabled optimization *overlaps* the one being priced, it inflates the delta.
  Two checks, both seconds: `strings` the plugin for the recognizer's accepted
  variants, then count claimed regions in the optimized HLO
  (`grep -o '"name":"<fusion>"' *after_optimizations.txt`) against the marker
  count in `before_optimizations.txt`. Note the marker *name* is absent from the
  optimized text whether it was claimed or inlined, so grepping for it there
  distinguishes nothing — the `__custom_fusion` backend config does. And the
  absence of an `INVALID_ARGUMENT: unknown variant` error proves only that the
  variant is *recognised*, never that it was *claimed*.
- **An ablation stand-in without `optimization_barrier` prices the whole
  downstream chain.** Replacing a transcript hop with same-shaped zeros makes
  the state a literal constant, so XLA folds every consumer that reads it and
  the arm is credited with work it did not remove — 138 regions removed against
  the 69 the fenced arm removes. Fence every stand-in whose output feeds a
  dependency chain, and sanity-check the arm with a static census before
  trusting its wall: the count of the thing you meant to keep (here, the fused
  rounds: 15 → 15) is the control that proves the arm prices the hop and not the
  pass.
- **On a shared box, gate on GPU residency, not on load average.** These pairings
  survive a busy *host*: one run measured cleanly at load 13.8 with 14 users. They
  do not survive a busy *card* — with 16 sibling `*_test_nvgpu_any` binaries
  resident, 13 of 15 pairs failed the control and baseline `open` swung
  30.42–98.88 ms on identical code. Poll
  `nvidia-smi --query-compute-apps=pid --format=csv,noheader` until it is empty
  for several consecutive samples (a bazel suite empties and refills between
  targets, so one clear sample is a false start). And report a run that keeps 2
  of 15 pairs as **unmeasured**, never as a null — a straddling IQR from two
  pairs is not a smaller version of the same result.
