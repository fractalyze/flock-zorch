# flock-zorch

A GPU prover for **flock**'s binary-field R1CS PIOP — the scheme from
[*Flock: Fast Proving for Batch Boolean Computations*](https://eprint.iacr.org/2026/1329)
(eprint 2026/1329) — built on Fractalyze's **zorch** stack. The whole prover is
authored once in Python/**FRX** (Fractalyze's JAX fork), and the compiler emits
the hardware code: the same readable source targets CPU and GPU, and its output
matches the reference flock prover bit-for-bit.

The point is a **single FRX/MLIR codebase, not a GPU rewrite of the proving
logic**. flock's prover is written as a clean statement of the math; FRX lowers
it to StableHLO/MLIR, and the compiler — carrying native finite-field dtypes
(`zk_dtypes`) and the carryless-multiply lowerings for GF(2¹²⁸) — compiles that
down to each target. The expensive field-arithmetic optimization lives in
compiler passes, out of the prover, and the byte-match gate guarantees those
transforms never change the output. The same program can shard across multiple
devices (GSPMD) without hand-written communication.

flock is an R1CS-over-GF(2¹²⁸) prover: two sumcheck PIOPs (zerocheck + lincheck)
over a Ligerito polynomial commitment, with a SHA-256 Fiat-Shamir
transcript, targeting hash-circuit statements (Keccak-f[1600], Keccak3, SHA-256,
BLAKE3). flock-zorch assembles that specific prover from zorch's scheme-agnostic
blocks (the round protocols and drivers, Fiat-Shamir, `Polynomial`, `PCS`,
fold, zero-check) and adds
only the flock-specific pieces the byte-match needs (GHASH-basis field, the
round-1 URM, the ∞-trick round loop, F128↔bytes serialization). The full prover
`prover.prove_fast` produces the complete `R1csProof` — commit → bind →
zerocheck → lincheck → batched dual-claim open, one shared challenger,
device-resident — reproducing flock `prove`'s proof bit-for-bit.

How the protocol is modelled — what a claim may state, which steps are roles
rather than functions, and what the tooling does not cover — is
[`docs/conventions.md`](docs/conventions.md).
How performance is measured — which numbers to trust, and the traps that have
each cost a session — is [`docs/measurement.md`](docs/measurement.md).

## Installation

**Python 3.11 on Linux x86_64, or macOS on Apple Silicon.** (`frxlib` ships a
cp311 wheel for those two platforms only — not 3.12/3.13, not Intel Macs.)

Run with `JAX_ENABLE_X64=true` — the GF(2¹²⁸) dtypes are 64-bit lane pairs and
x32 truncates them.

### CPU

```sh
pip install flock-zorch
```

### GPU (CUDA 12)

```sh
pip install flock-zorch 'frx[cuda12]' \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

The extra index carries the CUDA plugin wheels, which are too large for PyPI's
per-file limit. It is not needed for the CPU tier.

### Verify

```sh
JAX_ENABLE_X64=true python -c \
    "import frx, flock_zorch.prover; print(frx.devices()); print(flock_zorch.__version__)"
```

`[CpuDevice(id=0)]` means the CPU tier; a CUDA install prints the GPU devices.
Importing `flock_zorch.prover` rather than the package is deliberate: the package
`__init__` is a docstring, so a bare import stays green on an x32 interpreter and
on a `zk-dtypes` too old for the binary-field dtypes.

## Setup

Install the git hooks with both stages named. Plain `pre-commit install` wires
only the `pre-commit` stage, which leaves the commit-message linter inactive —
a malformed commit message then sails through to CI:

```sh
pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org):
a valid type, a lowercase summary with no trailing period, a header of at most
80 characters, and a body on everything but `docs`. The scope is the scheme the
change lives in — `hash`, `lincheck`, `pcs`, `sumcheck`, `zerocheck` — or one of
`verifier`, `ghash`, `prover`, `sha256_challenger`, `blake3_challenger`, `fs`,
`release` for the modules directly under the package. A change spanning several of them takes no scope.
The same linter runs in CI over every commit in a pull request and over the PR
title.

No submodules and nothing to clone by hand — both pinned deps are fetched by the
build:

| dep | how |
|---|---|
| **flock** — the reference prover + byte-compare oracle | a cargo **git rev dep** (`flock-core` / `flock-prover` in [`Cargo.toml`](https://github.com/fractalyze/flock-zorch/blob/main/Cargo.toml)); `cargo build` fetches it at the pinned rev, and `examples/dump_*.rs` drive it to dump the golden fixtures |
| **zorch** — the scheme-agnostic spine (the device Fiat-Shamir transcript, the `ProverRound`/`VerifierRound` protocols and their drivers, the `ProverStage`/`VerifierStage` claim-reduction roles, `pcs.ligerito`) | a bazel **`git_override`** in [`MODULE.bazel`](https://github.com/fractalyze/flock-zorch/blob/main/MODULE.bazel); bazel fetches it |
| **hash-frx** — the symmetric hash layer (`hash_frx.sha256` behind the Merkle leaves, `hash_frx.blake3` behind witness generation) | a pinned wheel in [`requirements.in`](https://github.com/fractalyze/flock-zorch/blob/main/requirements.in), from the Fractalyze package index |

**Prerequisites** — an NVIDIA GPU (CUDA; RTX 5090 / sm_120 reference), a Rust
toolchain (`flock-core` is edition 2024), Python 3.11. For the GPU fast path, a **CUDA 13.3
`ptxas`** first on `PATH` (`/usr/local/cuda` is not necessarily one): with it
the compiler emits the hardware `clmad` GF(2¹²⁸) multiply; without it, the
software `binary_field_ghash` multiply — same output, just slower. `PATH` — or
`CUDA_DIR` — is what selects it, **not `CUDA_ROOT`**, which frx overwrites at
import. See [`docs/measurement.md`](docs/measurement.md), which also covers why
a stale compilation cache can hide the fix. The `*_oracle_test.py` gates and
`prove_phase_bench.py` refuse to run on a toolchain that cannot assemble
`clmad`, so a mis-set environment fails loudly rather than as a phantom
regression.

```bash
git clone https://github.com/fractalyze/flock-zorch.git && cd flock-zorch
```

Reproduction has three tiers with independent deps: a **Rust toolchain**
regenerates the golden fixtures by driving the pinned flock (no GPU, no Python);
the **CPU byte-match** checks the frx port against them under **Bazel** (deps from
the pip lock, zorch from the git_override — no venv); the **GPU byte-match** runs
the port on-device from a **venv**. Build the venv once (the other two tiers need
nothing installed):

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.in --extra-index-url https://fractalyze.github.io/pypi/simple/
```

### Bumping the pins

- **flock** — bump the `rev` on the `flock-core` / `flock-prover` git deps in
  [`Cargo.toml`](https://github.com/fractalyze/flock-zorch/blob/main/Cargo.toml); cargo re-fetches on the next build.
- **zorch** — bump the `git_override` commit in [`MODULE.bazel`](https://github.com/fractalyze/flock-zorch/blob/main/MODULE.bazel),
  and move `requirements.in`'s `frx` / `frxlib` / `frx-cuda12` wheels to the SAME
  version as zorch's own `requirements.in` — the binary-field GPU kernels must
  match, and CPU-only CI can't catch a desync.

Then re-verify before pushing:

```bash
scripts/dump_goldens.sh core && bazel test //python:all
```

## Reproduce

The oracle is the pinned flock itself: `examples/dump_*.rs` dump fixtures from
`flock-core`, and the `*_oracle_test.py` gates byte-compare the FRX port's
serialized proofs against them. The gates are **proof-level**: every field of a
full serialized proof is compared, which transitively pins every layer under it
(FS framing, NTT, Merkle/octopus, zerocheck, lincheck, ring-switch — one
diverging byte anywhere flips every Fiat-Shamir draw after it). Primitives are
covered by python-native tests (no goldens); the retired per-layer golden gates
live in git history.

### Bazel tests (CPU)

Run under bazel — deps from the pip lock, `zorch` from the `MODULE.bazel`
`git_override`, goldens from `//artifacts` runfiles. One byte-match gate (the
full `LigeritoProof` — flock's fused prove has no config below m=22, so the e2e
gate can't come down to CPU) plus the native tests:

```bash
scripts/dump_goldens.sh core              # goldens the gates byte-compare against
bazel test //python:all                   # (JAX_PLATFORMS=cpu + x64 pinned in .bazelrc)
bazel test //python:ligerito_oracle_test  # the CPU byte-match anchor alone
```

### Proof gates (GPU, venv)

The full-prove gates — the identity e2e and the hash-circuit provers
(keccak/sha2/blake3, hundreds-of-MB goldens) — are **not** bazel targets (the
CUDA wheels aren't hermetic). Run them on the venv, resolving the same
git_override'd zorch via `scripts/zorch_pythonpath.sh`:

```bash
export FRX_PLATFORMS=cuda,cpu              # GPU prover + CPU SHA query chains
export FRX_ENABLE_X64=1                    # required by packed F128 witnesses
unset JAX_PLATFORMS JAX_ENABLE_X64         # avoid overriding the frx settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # don't grab ~75% of VRAM up front
export PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)"
CUDA13=/usr/local/cuda                       # MUST be a 13.3 toolchain
export PATH="$CUDA13/bin:$PATH"              # PATH selects clmad — NOT CUDA_ROOT
VENV=.venv/bin/python
scripts/dump_goldens.sh all                  # + the real hash circuits
$VENV python/flock_zorch/testing/e2e_ligerito_oracle_test.py    # fused prove (identity R1CS)
$VENV python/flock_zorch/testing/keccak3_ligerito_oracle_test.py # Keccak full prove (Ligerito)
$VENV python/flock_zorch/testing/blake3_ligerito_oracle_test.py
$VENV python/flock_zorch/testing/bench_ligerito_oracle_test.py  # snark.fast profile (BLAKE3 FS+Merkle) vs the challenge fork
```

The full proof-gate list is the `*_oracle_test.py` set under
`python/flock_zorch/testing/`. `artifacts/` is gitignored (regenerable, and
`blake3_golden.bin` alone is ~118 MB); `scripts/dump_goldens.sh [core|all]`
rebuilds it from the pinned flock.

### One benchmark point (SHA-256, m=26)

```bash
VENV=.venv/bin/python                                                                    # the venv from Setup
cargo run --release --example dump_sha2_ligerito -- 2048 artifacts/sha2_ligerito_golden.bin  # real R1CS, m=26
cargo build --release --example bench_sha2_ligerito_cpu                                   # CPU anchor
export FRX_PLATFORMS=cuda,cpu FRX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS JAX_ENABLE_X64
export PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)"
CUDA13=/usr/local/cuda                       # required for the hardware clmad path
export PATH="$CUDA13/bin:$PATH"              # PATH selects clmad — NOT CUDA_ROOT
CPU=$(target/release/examples/bench_sha2_ligerito_cpu 2048 | grep -oE '[0-9.]+ ms' | head -1)
$VENV python/flock_zorch/testing/prove_phase_bench.py sha2 --cpu-ms "${CPU%% ms}"         # GPU vs CPU
```

`prove_phase_bench.py` also splits the prove into commit / zerocheck / lincheck /
open and reports hashes/second, and refuses to print absolute numbers when
another process is using the GPU — a neighbour saturating the SMs inflates a warm
prove ~28× here, which is enough to invent a result. Swap `sha2` for `blake3` or
`keccak3`; `--golden` points it at an m-variant dump.

### The snark.fast harness window

The [flock-challenge](https://github.com/Layr-Labs/flock-challenge) benchmark
measures a fresh worker per trial — seed on stdin starts the clock, the proof
file's rename stops it — and verifies every proof with the fork's own verifier
(BLAKE3 Fiat-Shamir and BLAKE3 Merkle on the `flock-bench-v0` domain; the
`prove_fast` profile plumbing selects that arm). Point its `WORKER` positional
at `scripts/bench_worker.sh`: the harness clears the worker's env, so the shim
restores it and exec's `python/flock_zorch/testing/bench_worker.py`, whose
timed body is byte-gated against a fork-verified bundle by
`bench_ligerito_oracle_test.py` (regen: `cargo run --release --example
dump_bench_ligerito`) — the harness's `verified=true` is the same acceptance
the gate pins. The shim also wires a per-wheel `JAX_COMPILATION_CACHE_DIR`:
a respawned worker must absorb the multi-minute XLA compile inside the
harness's 300 s readiness budget, so warm trials have to hit that cache.

```bash
git -C "$SCRATCH" clone https://github.com/Layr-Labs/flock-challenge.git
git -C "$SCRATCH/flock-challenge" checkout d866043
# One-time compile-cache warm per wheel and log2, OUTSIDE the harness's
# readiness budget (a cold cache pays ~10 min of XLA compile):
echo 42 | scripts/bench_worker.sh 8 "$SCRATCH/warm.ready" "$SCRATCH/warm.proof"
cargo run --manifest-path "$SCRATCH/flock-challenge/Cargo.toml" --release \
  -p flock-benchmark-harness -- \
  "$PWD/scripts/bench_worker.sh" "$SCRATCH/bench" score.json summary.json \
  8 32 2 10   # LOG2 THREADS WARMUP_RUNS RUNS; log2 8 → m=22, 18 → m=32
```

## Benchmark

Apple-to-apple: **unmodified flock CPU vs flock-zorch GPU on the same idle
machine** (RTX 5090, Ryzen 9 9950X), same-instance both sides. Every instance is
a real flock hash-circuit R1CS at flock's shipped size, swept over witness size
to locate the crossover. GPU uses hardware `clmad`; timing is warm best-of-3
(JIT compile excluded), with the card verified idle. CPU rows use pinned flock
`85fc0e7`, thin LTO, one codegen unit, and `target-cpu=native`
(AVX-512/VPCLMULQDQ). GPU rows use zorch `cad4fea` and FRX
`0.10.1.dev20260803035606`.

**The GPU columns below are stale.** `cad4fea` was an orphaned-branch pin that
#199 dropped when it repinned to zorch main, and the repin alone moved m32
throughput substantially. Every GPU number in the tables predates that, so read
them as a floor, not as current performance.

### Keccak3 (Ligerito)

| m   | hash slots | flock CPU (ms) | GPU (ms) | Keccak/s | speedup   |
| --- | ---------- | -------------- | -------- | -------- | --------- |
| 22  | 96         | 6.54           | 9.01     | 10,653   | 0.73×     |
| 24  | 384        | 7.38           | 9.76     | 39,335   | 0.76×     |
| 26  | 1536       | 15.21          | 11.53    | 133,173  | **1.32×** |
| 28  | 6144       | 52.67          | 15.65    | 392,485  | **3.37×** |
| 30† | 24576      | 218.17         | 33.21    | 739,960  | **6.57×** |
| 31† | 49152      | 456.69         | 56.47    | 870,485  | **8.09×** |

† Rows marked † were measured under `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`,
believed at the time to be required against BFC fragmentation (#131). That is
not a general rule for large `m`: at m32 the default allocator runs clean in
both throughput and phase-split mode, and `cuda_async` *inflates* the prove
**~14%** (71.8 vs 81.6 ms, means of three fresh processes per arm).
Reach for it only if you actually hit an allocator OOM — and expect these rows
to understate the current prover until they are re-measured. See
[`docs/measurement.md`](docs/measurement.md).

### BLAKE3 (Ligerito)

Both sides exclude witness construction. CUDA 13.3 `ptxas` emitted PTX 9.3 for
sm_120 (driver 610.43.02).

| m   | n_comp | flock CPU (ms) | GPU (ms) | BLAKE/s   | speedup    |
| --- | ------ | -------------- | -------- | --------- | ---------- |
| 26  | 4096   | 14.58          | 9.45     | 433,299   | **1.54×** |
| 28  | 16384  | 44.77          | 13.78    | 1,188,901 | **3.25×** |
| 31† | 131072 | 367.18         | 55.46    | 2,363,383 | **6.62×** |

Reproduce all three GPU points with the shared goldens:

```bash
export FLOCK_ZORCH_ARTIFACTS="$PWD/artifacts"   # where dump_goldens.sh writes
export FRX_PLATFORMS=cuda,cpu FRX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# A 13.3 toolchain must come first on PATH — /usr/local/cuda is not necessarily
# one, and 12.9 silently selects the software GF(2¹²⁸) multiply (~5.5× at m28)
# and fails outright in nvlink on the larger opens. XLA resolves ptxas from
# CUDA_DIR then PATH; CUDA_ROOT plays no part (frx overwrites it at import), so
# probe the ptxas PATH actually resolves. Keep these two exports SEPARATE: in a
# single `export A=x PATH="$A/bin:$PATH"`, $A still expands to its OLD value,
# so the toolchain never reaches PATH.
CUDA13=/usr/local/cuda
export PATH="$CUDA13/bin:$PATH"
ptxas --version | grep -q 'release 13.3' ||
  echo 'WARNING: the ptxas on PATH is not 13.3 — the gates will refuse'
export PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)"
VENV=.venv/bin/python
# Needed when the host does not already provide CUDA 12 user-space libraries.
# Keep the version identical to requirements.in's frx pin.
.venv/bin/pip install "frx-cuda12-plugin[with-cuda]==$(sed -n 's/^frx==//p' requirements.in)" --extra-index-url https://fractalyze.github.io/pypi/simple/
$VENV python/flock_zorch/testing/prove_phase_bench.py blake3 --throughput --golden blake3_ligerito_golden_m26.bin --cpu-ms 14.58
$VENV python/flock_zorch/testing/prove_phase_bench.py blake3 --throughput --golden blake3_ligerito_golden_m28.bin --cpu-ms 44.77
$VENV python/flock_zorch/testing/prove_phase_bench.py blake3 --throughput --golden blake3_ligerito_golden_m31.bin --cpu-ms 367.18
```

Omit `--throughput` for the synchronized commit / zerocheck / lincheck / open
diagnostic breakdown; its phase barriers intentionally report a slower time.

## Acknowledgments

The proving scheme and the reference implementation are
[**flock**](https://github.com/succinctlabs/flock) by Succinct Labs — the
[flock paper](https://eprint.iacr.org/2026/1329) (eprint 2026/1329). flock-zorch
is an independent GPU implementation of that scheme on the zorch stack; the
unmodified `succinctlabs/flock` prover is pinned as the `flock-core` /
`flock-prover` git rev dep and is the byte-compare oracle every gate checks
against. All credit for the scheme and the R1CS PIOP design is theirs.
