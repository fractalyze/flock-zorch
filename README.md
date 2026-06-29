# flock-zorch

A GPU port of [succinctlabs/flock](https://github.com/succinctlabs/flock) — a
binary-field R1CS PIOP prover — onto Fractalyze's **zorch / ZKX** compiler stack.

The goal is to take flock's CPU prover and run it on the GPU **without forking the
proving logic**: the same readable Python expresses the math, and the compiler
emits the hardware code. Every layer is gated **byte-identical against unmodified
flock** (field → additive NTT → Merkle → zerocheck → lincheck → PCS → full
`R1csProof`), and the full prover reaches **≥10× the CPU** at production witness
sizes — peaking at **15.4×** on the same machine.

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

Apple-to-apple: **unmodified flock CPU vs flock-zorch GPU on the same machine**
(RTX 5090, Ryzen 9 9950X3D), GPU using the `clmad` carryless-multiply kernel,
byte-identity confirmed before every timing. Full machine spec, methodology, the
byte-identity table, and the software-`field.mul` comparison are in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

### End-to-end full prover (the number that matters)

`prover.prove_fast` — the complete `R1csProof` (commit → bind → zerocheck →
lincheck → batched dual-claim open, on one shared challenger, device-resident) —
reproduces flock `prove`'s proof **bit-for-bit** (`e2e_oracle_test.py`, m=13, all
39 fields), on an **identity R1CS**:

| m  | flock CPU (ms) | GPU clmad (ms) | speedup |
|----|----------------|----------------|---------|
| 22 | 72.2           | 63.2           | 1.1×    |
| 26 | 1,207.2        | 111.2          | **10.9×** |
| 28 | 4,742.8        | 308.9          | **15.4×** |

The GPU win **grows with m**: flock's sequential SHA-256 Fiat-Shamir chain (the
oft-feared critical path) does *not* dominate — the per-round host round-trips are
cheap against the bulk NTT / URM / FRI work the GPU crushes. Below ~m=22 the launch
overhead dominates and the CPU is competitive.

**On a real hash circuit, the honest picture is more nuanced.** Keccak3 Ligerito
at m=22 runs **0.10× (the GPU is ~10× *slower*)**: at that size there are many
sequential FS rounds and too little data-parallel work per round to amortize launch
overhead. The GPU advantage needs large witnesses (m≥26), exactly as the identity
curve shows. We report both rather than only the favorable case.

### Supporting micro-benchmarks (clmad, same box)

The dominant primitives, where the GPU's width + memory bandwidth pay off:

| primitive            | size sweep      | speedup vs flock CPU |
|----------------------|-----------------|----------------------|
| additive NTT         | log_d = 16 → 24 | 80× → **445×** (peak @22) |
| sumcheck `build_eq`  | n = 16 → 24     | 18× → **372×**       |

The `clmad` kernel is memory-bandwidth-bound (~1558 GB/s, the XOR-add ceiling on
this GPU), which makes the GF(2¹²⁸) multiply effectively free.

## Caveats (read these)

1. **zorch is early-stage.** It is in early bootstrap; this codebase is written to
   keep the **CPU prover readable while enabling GPU codegen** — i.e. optimization
   is delegated to the ZKX/MLIR compiler rather than hand-tuned kernels embedded in
   the prover. Some paths are still maturing (see #3).
2. **The CPU baseline is x86 *scalar*.** flock's NEON/parallel paths are
   `aarch64`-gated, so they don't compile on this x86 box. Apple silicon would
   narrow the gap; the definitive equivalence test wants flock built on a MacBook.
3. **Identity R1CS is dense/degenerate**, and a real sparse circuit lets the CPU
   hit fast-paths the generic dense prover skips — hence we report the real
   hash-circuit number alongside it. The SHA-256/BLAKE3 GPU full-prover numbers are
   not yet collected (a JAX interior-padding limitation in the CSC lincheck fold on
   this jaxlib version; Keccak provers use a different circuit and are unaffected).
4. **GPU wins scale with m** (m≥26). Don't read the headline as a universal speedup.

## Byte-identity (the correctness contract)

Nothing is "done" until its `*_oracle_test` is green on GPU against fixtures dumped
from unmodified flock-core (`examples/dump_*.rs`), so the gates transitively pin us
to upstream. Confirmed bit-for-bit on GPU:

| Layer | Backend |
|-------|---------|
| GF(2¹²⁸) multiply (GHASH basis), additive NTT (LCH), SHA-256, Merkle root | clmad / sw |
| sumcheck core (`build_eq` / `fold` / `round_pair`), Fiat-Shamir challenger | clmad |
| zerocheck `prove_packed`, lincheck `prove`, ring-switch + BaseFold + `pcs::open` | clmad + sw |
| **full fused prover** `prove_fast` → `R1csProof` (m=13–20) | clmad |
| **hash circuits** — Keccak-f[1600] · Keccak3 · SHA-256 · BLAKE3 (BaseFold + Ligerito) | clmad |

flock-zorch matches flock on *both* the proving scheme and the full hash-circuit
set. Architecture and the non-negotiable invariants are in [`CLAUDE.md`](CLAUDE.md).

## Setup

`flock` and `zorch` are pinned git submodules under `third_party/`. One command
bootstraps everything (submodules → venv → cargo build → goldens → smoke gates):

```bash
git clone --recursive git@github.com:fractalyze/flock-zorch.git && cd flock-zorch
scripts/setup.sh
```

Full instructions — prerequisites, the gate environment, golden regeneration, the
`clmad` GPU acceleration, and the benchmarks — are in [`docs/SETUP.md`](docs/SETUP.md).

### Reproduce the headline E2E benchmark

```bash
cargo build --release --example bench_e2e_cpu          # CPU anchor (flock-matched flags)
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=python:third_party/zorch
export FLOCK_CLMAD_CUBIN=$(pwd)/optim/clmad/ghash_mul.cubin
$VENV python/flock_zorch/testing/e2e_fused_bench.py 22 26 28
```

## Toolchain

- **GPU:** RTX 5090 (sm_120, Blackwell), CUDA 12.9+. zorch venv built by
  `scripts/setup.sh` (jax_fork jax-cuda12 stack + `jax-cuda12-pjrt` + `zk_dtypes`).
  `jax_enable_x64` is required for the uint64 field lanes.
- **Rust:** standalone rustup in `~/.cargo` (flock is edition 2024). `flock-core`
  is a path dep at `third_party/flock/crates/flock-core` — the byte-compare baseline.
