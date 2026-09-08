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
- **Against flock's `cuda-ghash/bench_ligerito`, drop our fold PoW first or the
  two provers are not doing the same work.** Their bench runs "grinding OFF",
  which means it calls `grind_pow(0)` — the unconditional 0-bit *query* grind —
  and performs **no fold grinds at all**. Our m32 golden carries
  `grinding_bits [0]*6`, identical to theirs, but
  `fold_grinding_bits [19, 14, 11, 8, 6, 4]`. Under
  `FlockChoreography.fold_grind_bits` (level `l`, fold round `j`, grinds
  `bits[l] - j` when > 0) that is **21 real searches**, every one of them inside
  `open` — the phase the cross-prover gap is largest on. It is easy to mistake
  for loop overhead: the count "21 grind whiles" is right, but they are not
  empty.

  **Count the hashes zorch evaluates, not the attempts the difficulty implies.**
  `grind_search` tests a whole `GRIND_WINDOW = 2^16` counter batch per
  `while_loop` step, so no grind costs less than one window however easy it is.
  That turns 1.07M expected attempts into **2^21 = 2.10M hashes actually
  evaluated**, and it relocates the work: level 0 is 97% of the attempts but
  only **53%** of the hashes, because 18 of the 21 grinds sit at ≤ 16 bits and
  each still pays a full window. Scoping a fix off the attempt count would aim
  at the wrong 18 searches. (0-bit grinds are exempt — the transcripts
  special-case them to the canonical zero witness, so the query grinds are free
  on both sides.)
  `prove_phase_bench --no-fold-grind` zeroes the fold schedule and leaves the
  query grinds, which lands both provers on the same work; **the proof is not
  gate-valid under that flag** (every challenge after a dropped grind moves), so
  it is a timing arm only. `rival_compare.py` runs both arms and prices the
  difference rather than assuming it: measured **+2.68 ms at m32**, of which
  +2.33 ms lands in `open`. Charging that to the prover gap is what turned a
  measured 4.9x `open` ratio into a reported 6.2x.

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

## The Yukon x86 ranked harness

Running `eigenlabs/flock-challenge-multi/x86` — the leaderboard the CPU tier is
measured against (#322, #323) — on the build box.

- **The frontier's sources need no Yukon CLI.**
  `Layr-Labs/flock-challenge-multi` keeps every submission as a branch, so the
  #1 submission is a plain clone: `submissions/<submission-id>`, whose tip
  commit message is `Validate submission <submission-id>`. The 1,633,567 comp/s
  frontier of 2026-09-08 is submission `e1a16581-883b-4459-8af3-bbdb3e0b2ea1`
  = `c75aece`, which is what `yukon clone` lands on. Worth knowing because the
  CLI is not installed on the build box and is not on PyPI; a session was spent
  looking for it. `yukon sync` takes no `--track` flag (it prints help).
- **`bwrap` must actually work, or every trial dies before it is measured.**
  Ubuntu ships `kernel.apparmor_restrict_unprivileged_userns=1`, which makes
  bubblewrap fail with `setting up uid map: Permission denied`; the sandboxed
  worker then exits 1, and the harness aborts with
  `worker exited before readiness with exit status: 1` before a single trial
  runs. Check it in one line — `bwrap --ro-bind / / /bin/true` — rather than
  reading the harness error as a submission bug: it reproduces identically with
  `FLOCK_NO_SEED_PIPE=1` and on an unmodified tree. The fix needs root
  (`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`, or an
  AppArmor profile for bwrap). Dropping the sandbox instead is a deviation from
  the ranked contract and changes what the number may be compared against, so it
  is a decision to take deliberately, not a workaround to reach for.
- **The x86 frontier's score is not all prover work — read
  `crates/flock-prover/src/seed_pipe.rs` before attributing the gap.** During
  the untimed warm-up the submission splices a pipe onto descriptor 0 and keeps
  the real stdin privately; a thread blocks on it, and when the seed arrives it
  proves from a closed-form parallel reimplementation of the harness's
  `generate_compressions` and publishes the bundle directly. That is legal under
  the contract — the seed is still read only after the harness starts its timer
  — but it removes the harness's strictly serial 6.5 M-draw input expansion from
  the timed window, which our worker still pays. Price that block separately
  before reading the remaining difference as prover speed.
- **A bare local run of that worker prints `missing seed on stdin` and exits 1
  while still publishing a valid, seed-dependent proof.** That is the protected
  main thread losing the stdin race to the seed-pipe thread, not a failed
  prove — the published bytes differ per seed and are the real proof. Do not
  read the exit code of a hand-run worker as the harness's verdict.
- **The multi fork's pristine flock is NOT the fork the bench gate pins.**
  `crates/flock-{core,prover}` on `flock-challenge-multi@main` differ from
  `Layr-Labs/flock-challenge@d866043`, the rev `Cargo.toml` pins as
  `flock-challenge-prover` and that `bench_ligerito_oracle_test.py` byte-gates
  against (the `DOMAIN` is the same `flock-bench-v0` in both). So the x86
  trusted verifier accepting a flock-zorch bundle is an open question, not a
  given; check it with a `BLAKE3_LOG2=8` smoke run before assuming the FRX arm
  can be scored by the x86 verifier rather than the Apple fork's harness.
