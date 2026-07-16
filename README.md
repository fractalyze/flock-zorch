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
over a BaseFold / Ligerito polynomial commitment, with a SHA-256 Fiat-Shamir
transcript, targeting hash-circuit statements (Keccak-f[1600], Keccak3, SHA-256,
BLAKE3). flock-zorch assembles that specific prover from zorch's scheme-agnostic
blocks (`Round`, Fiat-Shamir, `Polynomial`, `PCS`, fold, zero-check) and adds
only the flock-specific pieces the byte-match needs (GHASH-basis field, the
round-1 URM, the ∞-trick round loop, F128↔bytes serialization). The full prover
`prover.prove_fast` produces the complete `R1csProof` — commit → bind →
zerocheck → lincheck → batched dual-claim open, one shared challenger,
device-resident — reproducing flock `prove`'s proof bit-for-bit.

## Setup

No submodules and nothing to clone by hand — both pinned deps are fetched by the
build:

| dep | how |
|---|---|
| **flock** — the reference prover + byte-compare oracle | a cargo **git rev dep** (`flock-core` / `flock-prover` in [`Cargo.toml`](Cargo.toml)); `cargo build` fetches it at the pinned rev, and `examples/dump_*.rs` drive it to dump the golden fixtures |
| **zorch** — the scheme-agnostic spine (`zorch.hash.sha256`, the device Fiat-Shamir transcript, the `Round`/`Bridge`/`Stage` chain roles, `pcs.basefold`) | a bazel **`git_override`** in [`MODULE.bazel`](MODULE.bazel); bazel fetches it |

**Prerequisites** — an NVIDIA GPU (CUDA; RTX 5090 / sm_120 reference), a Rust
toolchain (`flock-core` is edition 2024), Python 3.11, and SSH access to
`fractalyze/zorch` (bazel clones it). For the GPU fast path, a **CUDA 13.3
`ptxas`** at `~/.local/cuda13/bin`: with it on `PATH` the pinned frx wheel's
compiler emits the hardware `clmad` GF(2¹²⁸) multiply; without it, the software
`binary_field_ghash` multiply — same output, just slower.

```bash
git clone git@github.com:fractalyze/flock-zorch.git && cd flock-zorch
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
  [`Cargo.toml`](Cargo.toml); cargo re-fetches on the next build.
- **zorch** — bump the `git_override` commit in [`MODULE.bazel`](MODULE.bazel),
  and move `requirements.in`'s `frx` / `frxlib` / `frx-cuda12` wheels to the SAME
  version as zorch's own `requirements.in` — the binary-field GPU kernels must
  match, and CPU-only CI can't catch a desync.

Then re-verify before pushing:

```bash
scripts/dump_goldens.sh core && bazel test //python:all
```

## Reproduce

The oracle is the pinned flock itself: `examples/dump_*.rs` dump fixtures from
`flock-core`, and each `*_oracle_test.py` checks the FRX port's serialized output
against them, anchored bottom-up (field → additive NTT → Merkle → zerocheck →
lincheck → PCS → full `R1csProof`). A layer is not done until its gate is green
on GPU.

### Core gates (bazel, CPU)

The 21 core gates run under bazel — deps from the pip lock, `zorch` from the
`MODULE.bazel` `git_override`, goldens from `//artifacts` runfiles:

```bash
scripts/dump_goldens.sh core           # goldens the gates byte-compare against
bazel test //python:all                # all 21 (JAX_PLATFORMS=cpu + x64 pinned in .bazelrc)
bazel test //python:e2e_oracle_test    # a single gate
```

### Heavy + GPU gates (venv)

The heavy hash-circuit gates (keccak/sha2/blake3 — hundreds-of-MB goldens) and
the GPU runs are **not** bazel targets (the CUDA wheels aren't hermetic). Run
them on the venv, resolving the same git_override'd zorch via
`scripts/zorch_pythonpath.sh`:

```bash
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # don't grab ~75% of VRAM up front
export PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)"
export PATH="$HOME/.local/cuda13/bin:$PATH"  # CUDA 13.3 ptxas -> compiler emits clmad
VENV=.venv/bin/python
scripts/dump_goldens.sh all                  # + the real hash circuits
$VENV python/flock_zorch/testing/e2e_oracle_test.py          # full prove on GPU
$VENV python/flock_zorch/testing/keccak_oracle_test.py       # Keccak full prove (BaseFold)
$VENV python/flock_zorch/testing/blake3_ligerito_oracle_test.py
```

The full per-layer + per-hash-circuit gate list is the `*_oracle_test.py` set
under `python/flock_zorch/testing/`. `artifacts/` is gitignored (regenerable, and
`blake3_golden.bin` alone is ~118 MB); `scripts/dump_goldens.sh [core|all]`
rebuilds it from the pinned flock.

### One benchmark point (SHA-256, m=26)

```bash
VENV=.venv/bin/python                                                       # the venv from Setup
cargo run --release --example dump_sha2 -- 2048 artifacts/sha2_golden.bin   # real R1CS, m=26
cargo build --release --example bench_sha2_cpu                              # CPU anchor
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)"
export PATH="$HOME/.local/cuda13/bin:$PATH"
CPU=$(target/release/examples/bench_sha2_cpu 2048 | grep -oE '[0-9.]+ ms' | head -1)
$VENV python/flock_zorch/testing/e2e_sha2_bench.py "${CPU%% ms}"            # GPU vs CPU
```

## Benchmark

Apple-to-apple: **unmodified flock CPU vs flock-zorch GPU on the same idle
machine** (RTX 5090, Ryzen 9 9950X), same-instance both sides. The golden is
dumped from flock-core, the CPU bench (`bench_*_cpu`, thin-LTO /
`codegen-units=1` / `target-cpu=native` — flock's honest x86 best) proves it, and
the GPU bench ingests the same golden. GPU uses the hardware `clmad` multiply;
timing is warm best-of-3 (JIT compile excluded), GPU verified idle. Every
instance is a real flock hash-circuit R1CS at flock's shipped size, swept over
the witness size m to locate the GPU/CPU crossover. The CPU baseline is x86
**scalar** (flock's NEON paths are aarch64-gated), so Apple silicon would shift
the crossover right. Measured on `main` (zorch `9cb08349`, FRX
`dev20260715063133`), 2026-07-16.

### Keccak3 (Ligerito) — crossover ≈ m=24

| m   | n_keccaks | flock CPU (ms) | GPU (ms) | speedup   |
| --- | --------- | -------------- | -------- | --------- |
| 22  | 49        | 27.0           | 47.2     | 0.57×     |
| 24  | 384       | 71.3           | 66.4     | **1.07×** |
| 26  | 1536      | 277.1          | 94.1     | **2.9×**  |
| 28  | 6144      | 1,167.1        | 185.7    | **6.3×**  |

The Ligerito open runs device-resident — zorch's recursive open compiles to one
device program (#479) and query positions are sampled on-device (#104) — so the
GPU is a slow-growing floor (47 → 186 ms) while the CPU is O(n): GPU wins from
m=24 and reaches 6.3× by m=28.

### SHA-256 (BaseFold) — crossover ≈ m=25

| m   | n_comp | flock CPU (ms) | GPU (ms) | speedup  |
| --- | ------ | -------------- | -------- | -------- |
| 24  | 512    | 63.8           | 76.1     | 0.8×     |
| 26  | 2048   | 222.7          | 111.7    | **2.0×** |
| 28  | 8192   | 928.7          | 243.7    | **3.8×** |

### BLAKE3 (BaseFold) — crossover ≈ m=25

| m   | n_comp | flock CPU (ms) | GPU (ms) | speedup  |
| --- | ------ | -------------- | -------- | -------- |
| 26  | 4096   | 276.2          | 123.9    | **2.2×** |
| 28  | 16384  | 1,070.5        | 237.6    | **4.5×** |

The BaseFold open drives the same device-native `zorch.pcs.basefold` seam the
Ligerito path uses, so its GPU floor stays low (76 → 244 ms) and it crosses over
at m≈25 like Ligerito — GPU 2.0× by m=26, 3.8× by m=28.

**Reading the numbers.** flock's prover is a sequential SHA-256 Fiat-Shamir
chain; at small m the per-round data-parallel work (NTT / URM / FRI) is too small
to amortize GPU launch overhead, so the CPU wins. The bulk work grows with m and
the GPU overtakes — at m≈24 for Ligerito, m≈25 for BaseFold — and the advantage
keeps growing above the crossover (Ligerito 6.3× and BaseFold 3.8–4.5× by m=28).
Reproduce any point with the
[SHA-256 recipe above](#one-benchmark-point-sha-256-m26) (swap `dump_sha2` /
`bench_sha2_cpu` / `e2e_sha2_bench.py` for the `blake3` / `keccak3_ligerito`
variants).

## Acknowledgments

The proving scheme and the reference implementation are
[**flock**](https://github.com/succinctlabs/flock) by Succinct Labs — the
[flock paper](https://eprint.iacr.org/2026/1329) (eprint 2026/1329). flock-zorch
is an independent GPU implementation of that scheme on the zorch stack; the
unmodified `succinctlabs/flock` prover is pinned as the `flock-core` /
`flock-prover` git rev dep and is the byte-compare oracle every gate checks
against. All credit for the scheme and the R1CS PIOP design is theirs.
