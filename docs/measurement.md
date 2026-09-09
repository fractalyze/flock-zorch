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
  **Answered 2026-09-08: it does accept one.** Driving the x86 trusted verifier
  at `scripts/bench_worker_cpu.sh` with `log2=8` reports `verified=true` on the
  warm-up and both measured trials, so the FRX arm can be scored by the x86
  verifier and the two arms are directly comparable.
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
- **The CPU tier's persistent XLA cache is a partial cache, and it warns about
  a machine mismatch against the very host that just wrote it.** Two things show
  up on a `FRX_PLATFORMS=cpu` worker on the 9950X, both harmless-looking and
  both budget-relevant:
  - `ToProto is not implemented for thunk kind: bit-reverse` — on a wheel
    predating fractalyze/xla#672 the persistent cache cannot *write*
    `jit_bit_reverse`, `jit__commit` or `jit__commit_prep`. Those programs are
    therefore recompiled in **every** fresh worker, so they are charged to the
    300 s readiness budget on all 120 trials, not just the first. Warming the
    cache does not make them go away; budget for them.
    **Fixed upstream 2026-09-08** by fractalyze/xla#672 (`00f941f5`), which
    registers `BitReverseThunk` with the CPU thunk serdes registry — but a fix
    in xla is not a fix in the wheel this repo pins, so probe the wheel before
    reading a warm-worker number. The CPU path's binary is
    `frxlib/libjax_common.so`, *not* the `xla_cuda_plugin.so` that
    [`development.md`](development.md)'s wheel-provenance rule names:

    ```sh
    B=.venv/lib/python3.11/site-packages/frxlib/libjax_common.so
    strings -a "$B" | grep -c BitReverseThunkProto   # the fix: 0 = absent
    strings -a "$B" | grep -c YnnFusionThunkProto    # control: must be > 0
    ```

    The control is the adjacent `oneof` case that predates the fix, so an empty
    first result means "absent" rather than "wrong binary". `dev20260826110348`
    gave 0 and 20 — absent; the pinned `dev20260908153807` gives 20 and 20, and
    a warm CPU worker on it logs no `ToProto is not implemented` line at all.
    Run the probe rather than reading xla's issue state: the fix landed in xla
    on 2026-09-08 while the wheel this repo pinned was still the 08-26 build,
    so "fixed upstream" and "fixed in the wheel you are measuring" were three
    days apart.
  - `cpu_aot_loader.cc: Target machine feature +prefer-no-gather is not
    supported on the host machine` on reload of entries this same box wrote
    minutes earlier. `prefer-no-gather` / `prefer-no-scatter` are Intel tuning
    pseudo-features that XLA records in the AOT result but the Zen 5 host
    feature query does not report, so the loader flags its own output as
    cross-machine. It is not a stale or shared cache (check the dir's mtimes
    before assuming it is), but the warning names SIGILL, so confirm what the
    loader actually does with a rejected entry before quoting a cached-path
    number.
- **The FRX CPU tier does reach a proof at m32 — but only on a wheel carrying
  fractalyze/xla#676.** The history is worth keeping, because the failure named
  the wrong culprit twice. On `frx==0.10.2.dev20260826110348`,
  `bench_worker.py` at `log2=18` died in the untimed warm-up prove ~75-109 s
  in, before it ever wrote the ready file:
  `Out of memory allocating 635655164048 bytes` — 592 GiB in one allocation,
  filed as fractalyze/xla#674. `--xla_dump_to` with
  `--xla_dump_hlo_pass_re=buffer-assignment` named the real module,
  `jit__round1_core` (the zerocheck round-1 URM), at 593.50 GiB, of which
  512 GiB was one `binary_field_ghash[16384,4096,64,8]` — the gf8 AES-basis
  expansion materialized whole. It was neither the machine nor the prover: the
  byte count was identical under `with-build-permit --mem 48` and `--mem 56`,
  so it was the compiled program's own buffer request. The cause was a fusion
  gap. `TreeReductionRewriter` rewrites a long-axis reduction into a chain of
  reduce-windows, and `CpuInstructionFusion` had never accepted a reduce-window
  as a fusion consumer, so the reduction's whole input was materialized;
  fractalyze/xla#676 lets a non-overlapping one take a fused producer. On the
  pinned `0.10.2.dev20260908153807`, which carries it, m32 runs end to end at
  45.3 GB peak RSS on this 60 GB box — see the m32 pair below — and the four
  other modules that peaked near 72 GB (`jit__slice_evals`,
  `jit__fold_packed_at_z`, `jit_rs_eq_ind`, `jit__partial_fold`) are no longer
  a wall either. Three things to carry forward:
  - **The host traceback named the wrong phase.** It pointed at
    `pcs/ligerito.py`'s `_packed_device_get`, which is only where the host
    blocked on an async error, and reads as an `open`-phase problem. Take the
    failing module from a buffer-assignment dump, never from the Python frame.
  - **It was size-gated, and nothing below m32 ever saw it.** m26 (`log2=12`)
    completed throughout. This is exactly the class
    [`development.md`](development.md) warns about: the standing gates run at
    m=22, so a path gated on size is never exercised by them — the m32 golden
    it tells you to gate with is what surfaced this.
  - **Measured on XLA:CPU only.** No tier has driven `bench_worker.py` at
    `log2=18` on the GPU backend, so whether it materialized the same
    intermediates was never tested; do not quote any of this as CPU-specific.

- **The m32 constants golden is ~2.2 GB and ~45 min to dump; it is not the
  86 MB one.** `bench_worker.py` loads `constants_golden(log2 + K_LOG)`, so the
  ranked `log2=18` needs `artifacts/blake3_ligerito_golden_m32.bin`, dumped with
  `cargo run --release --example dump_blake3_ligerito -- 262144 <out>`
  (`n_comp` is the first positional; the default 256 is the m22 file the gates
  use — do not overwrite it). Size is a fixed base plus a per-compression term,
  `≈ 84.6 MB + 8.21 KB * n_comp`, which the 256 / 1024 / 4096 points fit to
  within 0.02 %; dump time is linear in `n_comp` (3 s / 40 s at 256 / 4096).
  Both matter because the file is gitignored, so it is regenerated per machine,
  and every one of the 120 fresh workers reads it inside the readiness budget.

- **A harness trial's startup is XLA compile of ONE uncached program — not
  tracing, and not the cache being broken.** The harness spawns a fresh worker
  per trial, so every trial repays the whole per-process startup. Attribute it
  with `JAX_LOG_COMPILES=1` before blaming the prover or reaching for a pin
  bump. Measured spawn → ready file on a WARM cache (`FRX_PLATFORMS=cpu`,
  9950X, `taskset -c 0-15`, frx `dev20260908153807` with flock-zorch#328 in
  tree), against the harness's 300 s readiness budget:

  | size | warm startup | timed prove | XLA compile inside startup |
  |---|---|---|---|
  | m26 (`log2=12`) | 22.8-24.8 s | 1.35 s | 13.87 s over 95 programs |
  | m32 (`log2=18`) | 116.2 s | 87.8 s | — |

  At m26 four programs are the whole compile bill — `_open_jitted` 5.52 s,
  `_witness_blake3_xla` 4.93 s, `_mlv_sumcheck` 2.19 s, `prove_inf_product`
  0.72 s, with ~91 others at ~0.5 s together. Every one of them is a cache
  *hit*, so that 13.87 s is deserialization, not compilation, and no pin bump
  or tracing work touches it. The remaining ~10 s of the 24 s is imports,
  tracing, lowering, the 118 MB golden load and the untimed warm-up prove.

  - **`_open_jitted` enters the persistent cache only since flock-zorch#328;
    before it, it silently did not.** It recompiled in full in every worker —
    20.11 s x 120 trials ~= 40 min of a ranked run — with no
    `Error writing persistent compilation cache entry` logged and no matching
    entry in the dir. The cause was in frx rather than XLA (`_cache_write`
    skips a program carrying host callbacks), which is why fractalyze/xla#673
    closed with no XLA change: taking the query sampler directly when the open
    is on CPU (flock-zorch#328) removed the callback, and the entry appeared.
    Check it by name, never by timing — `ls "$JAX_COMPILATION_CACHE_DIR" |
    grep open` must show `jit__open_jitted-<hash>-cache`, and it is the largest
    entry in the dir. The same program now costs 5.52 s to load against 20.11 s
    to compile, which is most of the warm-startup step from 39.76 s to a
    23.8 s median.
  - **Do not credit that step to the wheel.** fractalyze/xla#672, which the
    same wheel carries, does what it says — the `ToProto` refusals go 2 -> 0
    and cold startup 77 s -> 37 s — but warm-to-warm it bought only **7.5 %**
    (42.98 -> 39.76 s). Startup is dominated by `_open_jitted`, and only
    flock-zorch#328 moved that. The two figures come from different sessions,
    so the direct evidence for the attribution is not the step itself but the
    same program's cost inside one measurement: 5.52 s loaded against 20.11 s
    compiled.
  - **The `cpu_aot_loader` machine-mismatch warnings are benign** — two per hit
    (`prefer-no-gather`, `prefer-no-scatter`), 190 for 95 hits, and the entry is
    then used. A rejected entry is not what is happening.
  - **Startup is NOT flat in `m` any more, and the old reading was an
    artifact.** It looked flat (30.5 s at m24 vs 36.6 s at m26) while
    `_open_jitted` recompiled from scratch in every worker and swamped
    everything size-dependent. With it cached, startup tracks how much
    executable has to be deserialized: 24 s at m26 against 116 s at m32.
  - **A trial costs startup plus prove, and nothing else worth counting.**
    Timed end to end, the harness runs 25.1 s/trial at m26 (175.8 s for 7)
    and 199.8 s/trial at m32 (599.4 s for 3) — against 25.2 s and 204.0 s for
    those two components alone, so its own spawn and verification overhead is
    inside the noise at both sizes. Do not budget a separate verification
    term. A ranked 120-trial run is therefore **~50 min at m26 and ~6.7 h at
    m32**; m32 acceptance is fractalyze/flock-zorch#325's, not a step inside a
    working session.
  - **AOT export (`jax.export`) is not the fix.** It targets tracing, a few
    seconds of the total.
  - Iterate **in-process** (a second prove is 2.0 s at m26); keep the harness
    for acceptance.
- **Do not time a hand-run worker as "ready file → process exit" — that is not
  the harness's window.** The harness stops its clock when the proof file is
  **renamed**; a process-exit timer additionally charges the write plus Python
  and XLA interpreter teardown, which measured ~2.0 s at m26 — i.e. it reported
  4.02 s for a prove the in-process timer puts at 2.0 s, a **2x** overstatement
  that turns straight into a 2x understatement of comp/s. Take the prove time
  from a second in-process `prove_bundle`, or from the harness's own
  `score.json`; never from wrapping the worker process.

### The m32 pair on build-server (2026-09-08/09) — the goal's scored instance

m32 (`log2=18`, 262,144 compressions) is the size the leaderboard scores, and
since the frx wheel carrying fractalyze/xla#676 both arms complete at it. Both
unsandboxed (#322 decision 6), `taskset -c 0-15`, 16 threads, `performance`
governor, box at 99 % idle:

| arm | measured | score | median | trials | verified |
|---|---|---|---|---|---|
| Yukon frontier `e1a16581` (`c75aece`) | 2026-09-08 | **1,138,183 comp/s** | 230 ms | 20 warm-up + 100 measured | 120/120 |
| flock-zorch FRX CPU tier | 2026-09-09 | **2,986 comp/s** (not ranked †) | 87.78 s | 3 warm-up + 10 measured | 13/13 |

**381x**, on this machine, at this size — the gap this goal has to close. Four
things qualify it, and none of them is "the run was noisy":

- † **The FRX arm is 13 trials, not the ranked 20/100, so it is not a ranked
  score and must never be quoted as one.** Its ten measured trials span
  87.0-88.3 s at p90/p10 = 1.013 — tighter than either arm's m26 run — so
  the median is well determined as a median; that is a different claim from a
  `score.json` under the leaderboard's contract. The ranked run costs ~6.7 h on
  this box (199.8 s/trial, timed end to end) and is owned by
  fractalyze/flock-zorch#325, whose end criteria already require it; it was
  dropped from #322's spec by decision 29 rather than left implicit.
- **The arms were measured a day apart** (frontier 2026-09-08, FRX 2026-09-09)
  on the same box under the same governor, pinning and sandbox decision. The
  frontier is a Rust binary and nothing on the wheel side touches it, but say
  so when quoting the pair rather than implying one sitting.
- **The ratio does not transfer to the leaderboard.** The same submission
  scores 1,633,567 on the official c7i.4xlarge against 1,138,183 here, so this
  is a ratio against a 1.44x-handicapped frontier.
- **Part of the frontier's margin is not prover work.** Its seed-pipe lifts the
  harness's strictly serial 6.5 M-draw input expansion out of the timed window,
  which our worker still pays. Price that block before reading 381x as prover
  speed.

The FRX arm's 116 s worker startup is **not** in that 381x: the harness starts
its clock when the seed is written and stops it at the proof rename, so startup
is charged only to the 300 s readiness budget (which it clears with 61 % to
spare) and to the wall-clock of a ranked run. Peak RSS is 45.3 GB against this
box's 60 GB — the only headroom figure in the pair that is close to a limit.

### The m26 pair on build-server (2026-09-08/09) — the dev-loop baseline

m26 (`log2=12`, 4096 compressions) is the fast loop the lever tasks iterate
against: a ranked 120-trial run costs ~50 min at m26 against ~6.7 h at m32. Both
arms under the ranked contract (20 warm-up discarded, 100 measured, median),
both unsandboxed (#322 decision 6), `taskset -c 0-15`, 16 threads,
`performance` governor, box at 99 % idle:

| arm | measured | score | median | p90/p10 | verified |
|---|---|---|---|---|---|
| Yukon frontier `e1a16581` (`c75aece`) | 2026-09-08 | **519,852 comp/s** | 7.879 ms | 3.80 | 120/120 |
| flock-zorch FRX CPU tier | 2026-09-09 | **3,022 comp/s** | 1.355 s | 1.15 | 120/120 |

**172x**, on this machine, at this size. Read with four caveats:

- **The arms were measured a day apart, not in one sitting.** Only the FRX arm
  was re-run for the wheel; the frontier row is the 2026-09-08 run verbatim,
  because the frontier is a Rust binary that nothing on the wheel side touches.
  Same box, same governor, same pinning, same sandbox decision — say so when
  quoting the pair. Its own dated row is in the frontier table below.
- **The FRX arm moved 1.56x since 2026-09-08** (median 2.117 s -> 1.355 s), and
  the tree moved by exactly two commits in between: #330 (zorch
  `3762ddce` -> `1d3db25f` and the frx quad to `dev20260908153807`) and #328.
  So the step belongs to that pair jointly — no single-commit attribution was
  measured, and the older figure was a 14-trial partial rather than a ranked
  run, so the two are not the same kind of number either.
- **The frontier's own m26 run is jittery: p90/p10 = 3.80**, against 1.048 at
  m32 and 1.284 at m24. Its trials are only ~7.9 ms, so worker spawn/teardown
  jitter dominates; one 44 ms outlier sits against a 7.1 ms p10. m26 flatters
  neither arm's precision, which is the price of using it as the fast loop —
  and it is why the m32 pair, not this one, is the number the goal is scored
  on.
- **The ratio is size-specific and machine-specific.** It is 172x here and
  381x at m32, because the FRX arm's fixed floors amortize with `m` while the
  frontier's do not; an earlier ~214x at m24 carried the process-teardown
  contamination described above, so do not read a trend across all three. And
  this 9950X runs the frontier 1.44x slower than the official c7i.4xlarge, so
  no ratio taken here transfers to the leaderboard.

### The x86 frontier measured on build-server (2026-09-08)

Submission `e1a16581` (`c75aece`) under the official harness, worker unsandboxed
(#322 decision B), box quiet at 99 % idle, `performance` governor:

| run | threads | score (comp/s) | median | p90/p10 |
|---|---|---|---|---|
| m32 (`log2=18`), default | 32 | 1,086,628 | 241 ms | 1.078 |
| m32, pinned to physical cores | 16 | **1,138,183** | 230 ms | 1.048 |
| m26 (`log2=12`), pinned | 16 | 519,852 | 7.879 ms | 3.80 |
| m24 (`log2=10`), default | 32 | 274,467 | 3.73 ms | 1.284 |

Four things to carry forward from it:

- **SMT costs ~4.7 % — pin to physical cores.** 16 threads on `taskset -c 0-15`
  beat all 32 SMT threads (1,138,183 vs 1,086,628) and were steadier
  (p90/p10 1.048 vs 1.078). Siblings are `N`/`N+16` here; confirm with
  `/sys/devices/system/cpu/cpu0/topology/thread_siblings_list` rather than
  assuming, since the pairing differs by machine.
- **Most of the leaderboard gap is hardware, not the prover.** The same
  submission scores 1,633,567 on the official c7i.4xlarge and 1,138,183 here —
  the 9950X is 1.44x slower at it. So an FRX-vs-frontier ratio taken on this box
  is a ratio against a 1.44x-handicapped frontier and does **not** transfer to
  the leaderboard; quote the machine with every ratio.
- **The `powersave` governor is not a threat on `amd-pstate-epp`.** Measured
  head to head at m24: `powersave`/`balance_performance` 278,087 vs
  `performance`/`performance` 274,467 — a 1.3 % delta in the wrong direction,
  i.e. noise. Unlike the old throttling governors, `amd-pstate-epp`'s
  `powersave` is full-range (cores were already boosting to 5.44 GHz), so a run
  under it does not need an asterisk. Do not spend a session getting root for
  this.
- **Load average measured *during* a run says nothing.** A 32-thread benchmark
  drives the 1-minute average to ~32 by construction, and it then decays for
  minutes afterwards. Judge quietness *before* starting, with instantaneous idle
  (`vmstat 2 3`), not with `uptime` during or after. Related: #322's Context
  discards an earlier 1,068,472 for having run at load 23-26 under `powersave`;
  the quiet `performance` run above lands within 1.7 % of it, so that figure was
  sound after all (different box of the same model, so machine and conditions
  are conflated in that one comparison).
