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
measured against (#322, #323) — on the build box. What any given run scored,
and on which day and which pins, belongs on those issues; what follows is the
part that stays true.

- **The frontier's sources need no Yukon CLI.**
  `Layr-Labs/flock-challenge-multi` keeps every submission as a branch, so a
  submission is a plain clone: `submissions/<submission-id>`, whose tip commit
  message is `Validate submission <submission-id>` — which is what makes the
  identity of a pulled frontier checkable rather than assumed. The CLI is
  neither installed on the build box nor on PyPI, so do not plan around it;
  `yukon sync` takes no `--track` flag (it prints help), and alone it restores
  the best promoted submission.
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
- **Unsandboxing the harness is one argument, not a patch — and it is the
  harness's own tested path.** `benchmark.sh` passes the trusted verifier a 9th
  positional (the sandbox scratch dir) only when `bwrap` is on `PATH`; with 8
  arguments the harness sets `Config.sandbox_scratch = None` and execs the
  worker bare, which `worker_command_runs_bare_without_sandbox` covers as a
  first-class case. So the local-dev path is reached by passing one fewer
  argument, never by hiding `bwrap` from `PATH` — a `PATH` shim is
  indistinguishable from sandbox evasion and will be refused by anything
  watching. Everything else (the `SHA256SUMS` check on the verifier, the locked
  offline candidate rebuild, the 20/100 trial contract) must still be run
  verbatim, or the number is not comparable at all. Whether to take that path is
  a decision, not a workaround — see the bullet above.
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
- **The multi fork's pristine flock is NOT the fork the bench gate pins, and
  the x86 trusted verifier accepts a flock-zorch bundle anyway.**
  `crates/flock-{core,prover}` on `flock-challenge-multi@main` differ from
  `Layr-Labs/flock-challenge@d866043`, the rev `Cargo.toml` pins as
  `flock-challenge-prover` and that `bench_ligerito_oracle_test.py` byte-gates
  against (the `DOMAIN` is the same `flock-bench-v0` in both). The divergence
  does not cost acceptance — driving the x86 trusted verifier at
  `scripts/bench_worker_cpu.sh` reports `verified=true` — so both arms can be
  scored by the same verifier and are directly comparable. Re-confirm with a
  `BLAKE3_LOG2=8` smoke run when either fork moves; it is seconds, and the
  alternative is discovering it 100 trials in.
- **A ratio between the two arms is a property of one machine and one size,
  and transfers to neither.** This box runs the frontier submission well below
  its score on the official c7i.4xlarge, so a ratio taken here is a ratio
  against a handicapped frontier and says nothing about the leaderboard. It
  also moves with `m`: the FRX arm carries fixed per-process floors that
  amortize as `m` grows while the frontier's trials are milliseconds, so the
  gap narrows with size. Measure at the size you intend to claim, and quote the
  machine with every ratio.
- **Both arms in one session, or say so.** The frontier is a Rust binary that
  no wheel or pin on our side touches, which makes re-using an earlier run of
  it tempting. That is allowed and often right — but an inherited arm is an
  inherited figure, so label it as one wherever the pair is quoted.
- **A reduced-trial median is not a ranked score.** The contract is 20 warm-ups
  discarded and 100 measured; anything else is a median, however tight its
  spread. Both are useful and they are not interchangeable — say which one a
  number is every time it appears, because the qualifier is what gets dropped
  when a figure is quoted onward.

### Machine conditions

- **SMT costs a few percent — pin to physical cores.** Half the threads on
  `taskset -c 0-15` beat all 32 SMT threads on this box and were steadier.
  Sibling pairing differs by machine, so read
  `/sys/devices/system/cpu/cpu0/topology/thread_siblings_list` rather than
  assuming `N`/`N+16`.
- **The `powersave` governor is not a threat on `amd-pstate-epp`.** Unlike the
  old throttling governors it is full-range — cores boost to their full
  frequency under it — and head-to-head against `performance` the difference
  is inside run-to-run noise. A run under it does not need an asterisk, so
  do not spend a session getting root to change it.
- **Load average measured *during* a run says nothing.** A 32-thread benchmark
  drives the 1-minute average to ~32 by construction, and it then decays for
  minutes afterwards. Judge quietness *before* starting, with instantaneous
  idle (`vmstat 2 3`) — a figure discarded for "the box was loaded" on the
  strength of a during-run load average is usually a sound figure thrown away.

### The FRX CPU tier under that harness

- **An OOM in the untimed warm-up naming a preposterous single allocation is
  the compiled program's own buffer request, not the machine's limit.**
  `bench_worker.py` dying before it writes the ready file with
  `Out of memory allocating <n> bytes`, where `n` is orders of magnitude past
  the box's RAM, means some program is materializing an intermediate whole —
  so the question is which one and why it was not fused, never how much RAM
  the box has. Three things pin it down, and each has been read wrong at
  least once:
  - **The host traceback names the wrong phase.** It points at the frame where
    the host blocked on an async error — `pcs/ligerito.py`'s
    `_packed_device_get`, which reads as an `open`-phase problem — not at the
    module that asked. Take the failing module from `--xla_dump_to` with
    `--xla_dump_hlo_pass_re=buffer-assignment` and sum the `allocation N: size`
    lines; never from the Python frame.
  - **It is not the memory cap.** The byte count is identical under different
    `with-build-permit --mem` values, because it is the compiled program's own
    buffer request. If tightening or loosening the cap moves it, you are
    looking at something else.
  - **It is size-gated, so the standing gates cannot see it.** Those run at
    m=22 while the wall appears only at the ranked size. This is exactly the
    class [`development.md`](development.md) warns about, and the m32-variant
    golden it tells you to gate with is what surfaces it. Reproduce the fix at
    a size that fits and confirm at the ranked one.

  The instance behind these rules was `CpuInstructionFusion` declining the
  reduce-windows `TreeReductionRewriter` emits as fusion consumers, so a
  reduction's whole input was materialized. Whether a GPU backend materializes
  the same intermediates has not been tested, so do not read any of this as
  CPU-specific.
- **The m32 constants golden is ~2.2 GB; it is not the 86 MB one.**
  `bench_worker.py` loads `constants_golden(log2 + K_LOG)`, so the ranked
  `log2=18` needs `artifacts/blake3_ligerito_golden_m32.bin`, dumped with
  `cargo run --release --example dump_blake3_ligerito -- 262144 <out>`
  (`n_comp` is the first positional; the default 256 is the m22 file the gates
  use — do not overwrite it). Size is a fixed base plus a per-compression term,
  `≈ 84.6 MB + 8.21 KB * n_comp`, which the 256 / 1024 / 4096 points fit to
  within 0.02 %; dump time is linear in `n_comp` and runs to tens of minutes at
  the ranked size. Both matter because the file is gitignored, so it is
  regenerated per machine, and every fresh worker reads it inside the readiness
  budget.
- **A harness trial's startup is XLA compile of a handful of programs — not
  tracing, and not the cache being broken.** The harness spawns a fresh worker
  per trial, so every trial repays the whole per-process startup against the
  300 s readiness budget. Attribute it with `JAX_LOG_COMPILES=1` before blaming
  the prover or reaching for a pin bump. On a warm cache at m26 the compile
  bill is ~14 s over ~95 programs, and four of them are nearly all of it
  (`_open_jitted`, `_witness_blake3_xla`, `_mlv_sumcheck`,
  `prove_inf_product`); the rest of the startup is imports, tracing, lowering
  and the golden load. Every one of those compiles is a cache *hit*, so what
  the budget buys is deserialization, which is why neither a pin bump nor AOT
  export (`jax.export` targets tracing, a few seconds of the total) moves it.
  - **A program the cache silently refuses is the thing to look for, and only
    a by-name check finds it.** Two mechanisms drop entries without an error:
    a thunk kind the CPU serdes registry does not know refuses the *write*
    with `ToProto is not implemented for thunk kind: <kind>`, and frx's
    `_cache_write` skips a program carrying host callbacks with no log line at
    all. Either way the program recompiles in full in every worker, which at
    120 trials is the difference between a working session and an overnight
    one. `ls "$JAX_COMPILATION_CACHE_DIR"` and confirm the expensive programs
    are there by name — `jit__open_jitted-<hash>-cache` above all, since it is
    the largest entry when present. Timing cannot distinguish a slow hit from
    a miss; the directory listing can.
  - **`cpu_aot_loader` machine-mismatch warnings on Zen 5 are benign.**
    `Target machine feature +prefer-no-gather is not supported on the host
    machine` (and `-no-scatter`) appear twice per hit on entries the same box
    wrote: they are Intel tuning pseudo-features XLA records in the AOT result
    that the host feature query does not report. The entry is used and the
    compile finishes in under a millisecond. The warning names SIGILL, which
    is what makes it look load-bearing.
  - **Startup is not flat in `m`.** Once the expensive programs are cached,
    startup tracks how much executable has to be deserialized, so it grows
    steeply with size — m32's is several times m26's. Dropping a size step is
    therefore a real way to shorten a run, and an assumption of flatness is a
    way to under-budget one by hours.
  - **A trial costs startup plus prove and nothing else worth counting.**
    Timed end to end, a harness trial matches those two components to within
    noise at both m26 and m32, so the harness's own spawn and verification
    overhead needs no separate term. Budget a ranked 120-trial run as
    `120 * (startup + prove)`; at the ranked size that is hours, which makes it
    its own dispatch rather than a step inside a working session.
  - Iterate **in-process**: a second `prove_bundle` skips the whole startup.
    Keep the harness for acceptance.
- **Do not time a hand-run worker as "ready file → process exit" — that is not
  the harness's window.** The harness stops its clock when the proof file is
  **renamed**; a process-exit timer additionally charges the write plus Python
  and XLA interpreter teardown, which at m26 is enough to report a prove as
  roughly twice its real cost, and that turns straight into a halving of
  comp/s. Take the prove time from a second in-process `prove_bundle`, or from
  the harness's own `score.json`; never from wrapping the worker process.

### The FRX CPU tier's phase split (#324)

Where the m26 prove's wall goes, and — more useful — which of it moves under
which repository. Produced by `cpu_phase_split.py`, which drives
`_bench_profile.prove_bundle` (the body the harness times, `serialize`
included) rather than `prove_phase_bench`'s witgen→open sha256 scope. The
absolute numbers belong to #324 and this box; the rules below are what stays
true.

- **The CPU ranking is not the GPU ranking, and it is not close.** On CUDA
  `open` is 55–58% of the prove and on Metal the top three phases tie (#292);
  on CPU at m26 `zerocheck` is **62%**, `lincheck` 24%, `open` 12%, and
  `commit` + `witgen` + `serialize` together under 3%. A lever ranked from the
  GPU split aims at the wrong phase. Re-rank per backend; do not carry one
  over.
- **A phase name does not name a lever — the kernel class does.** `zerocheck`
  is three different pieces of work under three different owners: a
  binary-field select + XOR-reduce (44% of op busy), GHASH multiplies (21%),
  and NTT transforms (11%). Splitting only to the phase would have sent the
  next round at "zerocheck" without saying which of the three to touch.
- **The same fusion name means different things in different modules.**
  `select_reduce-window_fusion` is the round-1 URM's
  `binary_field_ghash[.,8,64]` select + XOR-reduce inside `_round1_core`, and
  a `u32[32]` proof-of-work reduce inside `_open_jitted` — 470 ms against
  under a millisecond. Any op-level table must key on (module, op); keying on
  the op name alone silently merges them. Read the class off
  `--xla_dump_to`'s `after_optimizations` HLO, never off the name.
- **Rank CPU work by parallel scaling as well as by share, because the two
  disagree.** A `taskset` sweep over 1/2/4/8/16 physical cores separates
  classes that are already parallel from classes that are not, and the second
  group is where the cheap wins are:

  | class | share at 16c | speedup 1c→16c |
  |---|---:|---:|
  | select + XOR-reduce (round-1 URM, Ligerito fold) | 47% | 12.8x |
  | GHASH multiply | 20% | 5.3x |
  | NTT transform | 10% | 6.4x |
  | lincheck segment fold | 9% | 1.9x |
  | FFI custom calls | 3% | 1.0x |

  The largest class is also the best-parallelized one; the whole prove scales
  8.2x on 16 cores (51% utilization), so roughly half the machine is idle for
  reasons that have nothing to do with the biggest kernel. **`--xla_cpu_...`
  parallelism work and binary-field lowering work are therefore separate
  levers, and the share column alone would have hidden the first.**
- **Trace with `XLA_FLAGS=--xla_cpu_enable_xprof_traceme=true`; it puts
  `hlo_op` / `hlo_module` on every op event, which is the only per-kernel
  attribution the CPU backend offers here.** Two consequences: the flag is in
  the compilation-cache key, so the first traced run recompiles everything
  (~78 s vs ~21 s of startup on a warm cache) — budget it, do not read it as a
  cache bug. And tracing inflates the *wall* ~3x while leaving *op busy*
  within ~2% of a clean run, so shares come from the trace and absolutes never
  do.
- **`perf` cannot count DRAM bytes on this box** —
  `kernel.perf_event_paranoid` is 4, so CPU events need `CAP_PERFMON`. That is
  one sysctl away, exactly like ncu's `sudo` above; ask for it rather than
  substituting a shape-derived byte model, which errs toward manufacturing
  headroom. Until then the core-count sweep is the discriminator that needs no
  counters: near-linear scaling rules out a bandwidth roof, and a class that
  stops scaling names one.
- **Startup is not the prove and the harness charges both.** In-process
  startup (golden load + warm-up prove on a warm cache) is ~21 s at m26 against
  a ~1.35 s prove, so a fresh-worker trial is dominated by deserialization.
  Quote them separately or a per-hash number becomes a statement about the
  compile cache.
