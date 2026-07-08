# flock-zorch

A GPU port of [succinctlabs/flock](https://github.com/succinctlabs/flock) — a
binary-field R1CS PIOP prover — onto Fractalyze's **zorch / ZKX** compiler stack.

The goal is to take flock's CPU prover and run it on the GPU **without forking the
proving logic**: the same readable Python expresses the math, and the compiler
emits the hardware code. Every layer is gated **byte-identical against unmodified
flock** (field → additive NTT → Merkle → zerocheck → lincheck → PCS → full
`R1csProof`), on flock's own hash-circuit instances. On the performance side the
GPU **crosses over the CPU at large witness sizes** (m ≳ 24 for BaseFold) and
loses below it — measured on flock's real circuits, not a synthetic one.

> flock is an R1CS-over-GF(2¹²⁸) prover: two sumcheck PIOPs (zerocheck + lincheck)
> over a BaseFold / Ligerito polynomial commitment, with a SHA-256 Fiat-Shamir
> transcript. It targets hash-circuit statements (Keccak-f[1600], Keccak3,
> SHA-256, BLAKE3).

## What is zorch?

[`zorch`](https://github.com/fractalyze/zorch) is **JAX-native building blocks for
modern SNARKs**. A modern SNARK is *IOP + PCS*; zorch provides those as composable
blocks (`Round`, Fiat-Shamir, `Polynomial`, `PCS`, fold, zero-check) the way a deep
learning stack composes layers. flock-zorch assembles flock's specific prover from
this scheme-agnostic spine and adds only the byte-identity-critical flock pieces
(GHASH-basis field, the round-1 URM, the ∞-trick round loop, F128↔bytes
serialization).

**Why a single JAX/MLIR codebase is the point — not just a GPU rewrite:**

- **One source, many backends.** The prover is written once in Python/JAX. JAX
  lowers it to **StableHLO/MLIR**, and ZKX (Fractalyze's XLA fork, with native
  finite-field dtypes) compiles that MLIR down to the target hardware. The *same*
  prover source targets CPU and GPU today, and other accelerators as ZKX grows —
  no reimplementation of the proving logic per device.
- **Acceleration lives in the compiler, not the prover.** The expensive
  field-arithmetic optimizations (e.g. carryless-multiply lowerings for GF(2¹²⁸))
  are MLIR passes, kept out of the prover code. The prover stays a clean statement
  of the math; the byte-match gate guarantees the compiler's transforms never
  change the output.
- **Parallelism for free.** Because the program is JAX/XLA, GSPMD **SPMD
  partitioning** can shard the same program across multiple devices (data / model
  partitioning) without hand-written communication — multi-GPU from one codebase.

## Benchmarks

Apple-to-apple: **unmodified flock CPU vs flock-zorch GPU on the same idle machine**
(RTX 5090, Ryzen 9 9950X3D), GPU using the `clmad` carryless-multiply kernel,
**byte-identity confirmed before every timing**. Every instance below is a *real
flock hash-circuit R1CS* at the size flock itself ships, swept over the witness
size m to locate the GPU/CPU crossover. Methodology, machine spec, and the
byte-identity table are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

The full prover is `prover.prove_fast` — the complete `R1csProof` (commit → bind →
zerocheck → lincheck → batched dual-claim open, one shared challenger,
device-resident) — which reproduces flock `prove`'s proof **bit-for-bit**.

### SHA-256 (BaseFold) — crossover ≈ m=24

| m  | flock CPU (ms) | GPU clmad (ms) | speedup |
|----|----------------|----------------|---------|
| 18 | 11.5           | 51.9           | 0.22×   |
| 20 | 13.8           | 51.0           | 0.27×   |
| 22 | 24.3           | 59.8           | 0.41×   |
| 24 | 67.0           | 66.6           | **1.01×** |
| 26 | 256.8          | 94.5           | **2.7×** |

### BLAKE3 (BaseFold)

| m  | flock CPU (ms) | GPU clmad (ms) | speedup |
|----|----------------|----------------|---------|
| 26 | 280.3          | 96.0           | **2.9×** |

### Keccak3 (Ligerito) — crossover ≈ m=27

| m  | flock CPU (ms) | GPU clmad (ms) | speedup |
|----|----------------|----------------|---------|
| 22 | 26.1           | 254.3          | 0.10×   |
| 24 | 70.4           | 277.7          | 0.25×   |
| 26 | 269.5          | 401.3          | 0.67×   |
| 28 | 1,118.2        | 627.0          | **1.8×** |

**Reading the numbers.** flock's prover is a sequential SHA-256 Fiat-Shamir chain;
at small m the per-round data-parallel work (NTT / URM / FRI) is too small to
amortize GPU launch overhead, so the CPU wins. The bulk work grows with m and the
GPU overtakes — at **m ≈ 24 for the BaseFold circuits** (SHA-256, BLAKE3) and later,
**m ≈ 27 for Ligerito** (Keccak3), whose extra recursive rounds add sequential
structure. Above the crossover the GPU advantage keeps growing with m. (The Keccak3
m=28 point runs near the RTX 5090's 32 GB ceiling — XLA hit a 16 GB allocation
retry — so it is memory-bound, not compute-bound.)

## Caveats (read these)

1. **zorch is early-stage.** It is in early bootstrap; this codebase is written to
   keep the **CPU prover readable while enabling GPU codegen** — i.e. optimization
   is delegated to the ZKX/MLIR compiler rather than hand-tuned kernels embedded in
   the prover. The crossover sits at large m precisely because the compiler-side
   field-arithmetic and fusion work is still maturing.
2. **The CPU baseline is x86 *scalar*.** flock's NEON/parallel paths are
   `aarch64`-gated, so they don't compile on this x86 box. Apple silicon would
   shift the crossover to the right; the definitive equivalence test wants flock
   built on a MacBook.
3. **The GPU only wins above the crossover** (m ≳ 24 BaseFold, ≳ 27 Ligerito). For
   the smaller instances the CPU is faster — don't read the large-m numbers as a
   universal speedup.

## Byte-identity (the correctness contract)

Nothing is "done" until its `*_oracle_test` is green on GPU against fixtures dumped
from unmodified flock-core (`examples/dump_*.rs`), so the gates transitively pin us
to upstream. Confirmed bit-for-bit on GPU:

| Layer | Backend |
|-------|---------|
| GF(2¹²⁸) multiply (GHASH basis), additive NTT (LCH), SHA-256, Merkle root | clmad / sw |
| sumcheck core (`build_eq` / `fold` / `round_pair`), Fiat-Shamir challenger | clmad |
| zerocheck `prove_packed`, lincheck `prove`, ring-switch + BaseFold + `pcs::open` | clmad + sw |
| **full fused prover** `prove_fast` → `R1csProof` | clmad |
| **hash circuits** — Keccak-f[1600] · Keccak3 · SHA-256 · BLAKE3 (BaseFold + Ligerito) | clmad |

flock-zorch matches flock on *both* the proving scheme and the full hash-circuit
set. Run the core gates with `bazel test //python:all`; the heavy hash-circuit
and GPU gates run on the venv — see [`docs/SETUP.md`](docs/SETUP.md).

## Setup

`flock` is a pinned git submodule under `third_party/` (the byte-compare oracle);
`zorch` is pinned via a bazel `git_override` in `MODULE.bazel` (bump = edit the
commit there, keeping `requirements.in` in lockstep). One command bootstraps
everything (submodules → venv → cargo build → goldens → smoke gates):

```bash
git clone --recursive git@github.com:fractalyze/flock-zorch.git && cd flock-zorch
scripts/setup.sh
```

Full instructions — prerequisites, the gate environment, golden regeneration, the
`clmad` GPU acceleration, and the benchmarks — are in [`docs/SETUP.md`](docs/SETUP.md).

### Reproduce a benchmark point (SHA-256, m=26)

```bash
VENV=.venv/bin/python                                                       # built by scripts/setup.sh
cargo run --release --example dump_sha2 -- 2048 artifacts/sha2_golden.bin   # real R1CS, m=26
cargo build --release --example bench_sha2_cpu                              # CPU anchor
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)"   # zorch from the MODULE.bazel git_override
export PATH="$HOME/.local/cuda13/bin:$PATH"                 # CUDA 13.3 ptxas → compiler emits clmad
CPU=$(target/release/examples/bench_sha2_cpu 2048 | grep -oE '[0-9.]+ ms')   # flock CPU ms
$VENV python/flock_zorch/testing/e2e_sha2_bench.py "${CPU%% ms}"             # GPU vs CPU
```

## Toolchain

- **GPU:** RTX 5090 (sm_120, Blackwell). JAX runtime is the **CUDA 12** stack
  (jax_fork jax-cuda12 + `jax-cuda12-pjrt` + `zk_dtypes`), built into `.venv` by
  `scripts/setup.sh`; hardware `clmad` is compiler-emitted when a **ptxas ≥ 13.3**
  is on `PATH` (sm_120 requires it). `jax_enable_x64` is required for the uint64 field lanes.
- **Rust:** standalone rustup in `~/.cargo` (flock is edition 2024). `flock-core`
  is a path dep at `third_party/flock/crates/flock-core` — the byte-compare baseline.
