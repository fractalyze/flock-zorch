# flock-zorch

GPU port of [succinctlabs/flock](https://github.com/succinctlabs/flock) — an
R1CS-over-GF(2) PIOP prover — onto Fractalyze's zorch / zkx compiler stack,
mirroring `bellman-zorch` / `accumulation-zorch`. See `CLAUDE.md` for the
architecture and non-negotiables.

## Setup (fresh GPU box)
`flock` and `zorch` are pinned as git submodules under `third_party/`. One command
bootstraps everything (submodules → venv → cargo build → goldens → smoke gates):

```bash
git clone --recursive git@github.com:fractalyze/flock-zorch.git && cd flock-zorch
scripts/setup.sh
```

Full instructions — prerequisites, the gate environment, golden regeneration, the
optional clmad GPU acceleration, and the apple-to-apple benchmarks — are in
[`docs/SETUP.md`](docs/SETUP.md).

## Status
Bottom-up port, each layer gated by a byte-match vs unmodified flock:

| Layer | State | Gate |
|-------|-------|------|
| GF(2¹²⁸) multiply (GHASH basis) | ✅ GPU byte-match, scales to 2²³ | `field_oracle_test.py` |
| additive NTT over F128 (LCH) | ✅ GPU byte-match (self-computed twiddles) | `ntt_oracle_test.py` |
| **GPU ≥10× CPU gate (additive NTT)** | ✅ **133–456× vs unmodified flock, byte-identical** | `cpu_vs_gpu.py` |
| SHA-256 (Merkle / challenger hash) | ✅ GPU byte-match (CPU-favorable, see below) | `sha256_oracle_test.py` |
| SHA-256 Merkle tree | ✅ GPU byte-match root | `merkle_oracle_test.py` |
| **PCS commit (pack→NTT→Merkle→root)** | ✅ **byte-identical root; encode 20–383× CPU** | `commit_oracle_test.py` |
| **sumcheck core (build_eq / fold / round_pair)** | ✅ **GPU byte-match (sw + clmad); build_eq 20–174× CPU** | `sumcheck_oracle_test.py`, `sumcheck_gpu_vs_cpu.py` |
| **Fiat-Shamir challenger (SHA-256 duplex)** | ✅ **byte-identical to flock `FsChallenger`** (on `zorch.byte_transcript`) | `challenger_oracle_test.py` |
| F8 layer + round-1 URM (F8 NTT / φ₈ / `round1_naive`) | ✅ GPU byte-match (φ₈ table + URM) | `gf8_urm_oracle_test.py` |
| **zerocheck `prove_packed` (full PIOP → ZerocheckProof)** | ✅ **byte-identical to flock** (sw + clmad), m=13/14 | `zerocheck_oracle_test.py` |
| **lincheck `prove` (2nd PIOP → LincheckProof)** | ✅ **byte-identical to flock** (sw + clmad), 6 regimes | `lincheck_oracle_test.py` |
| ring-switch + BaseFold + **full `pcs::open`** | ✅ **byte-identical to flock** (sw + clmad) | `ring_switch_oracle_test.py`, `basefold_oracle_test.py`, `pcs_open_oracle_test.py` |
| **e2e fused prover (`prover.prove_fast`)** | ✅ **byte-identical to flock `prove`** (full `R1csProof`: commit→bind→zerocheck→lincheck→batched dual-claim open), m=13–20 | `e2e_oracle_test.py` |
| **hash-circuit R1CS provers (real instances)** | ✅ **Keccak-f[1600] · Keccak3 · SHA-256 · BLAKE3 all byte-identical to flock** (full `R1csProof`; BaseFold + Ligerito) — same proving scheme + same hash set as flock | `keccak_*`, `keccak3_ligerito_*`, `sha2_*`/`sha2_ligerito_*`, `blake3_oracle_test.py` / `blake3_ligerito_oracle_test.py` |
| **GPU full-prover ≥10× CPU** | ✅ **9.5× @m=26, 16.6× @m=28** (vs same identity-R1CS x86 CPU) | `e2e_fused_bench.py` + `bench_e2e_cpu.rs` |
| host SHA-NI Merkle FFI (SHA off-GPU) | ✅ byte-identical, merkle 35→1ms | `merkle_oracle_test.py` (`FLOCK_HOST_SHA=1`) |
| fused `.mlirbc` + Rust host (PJRT) | ⏳ optional (prover runs as host-driven jax today) | — |

### BLAKE3 hash circuit (closes the last coverage gap vs flock)
flock supports four hash circuits — Keccak-f[1600], Keccak3, SHA-256, BLAKE3.
The first three were already byte-identical; **BLAKE3 now is too**, so flock-zorch
matches flock on *both* the proving scheme *and* the full hash-circuit set. BLAKE3's
R1CS mirrors sha2 (populated `a_0`/`b_0` sparse matrices via `build_matrices`,
folded by the generic CSC lincheck circuit), so the port reuses the entire generic
GPU prover with **no blake3-specific Python** — only a golden dumper
(`examples/dump_blake3*.rs`) and the byte gates. Verified bit-for-bit vs flock:
- BaseFold (`blake3_oracle_test.py`, m=22): full `R1csProof` ✅
- Ligerito (`blake3_ligerito_oracle_test.py`, m=22): full `R1csProofLigerito` ✅

**The CSC lincheck fold is now on device.** It was the bottleneck — a host
`np.bitwise_xor.at` over BLAKE3's ~21 M nonzeros at **491 ms**, a fixed per-block
tax that pinned the end-to-end prove at 555 ms (0.1×). The transposed binary matvec
`out[c] = Σ_{r:M[r,c]=1} eq[r]` is a **column-segmented XOR-reduce**; a padded gather
blows up on the skewed const_pin column (~15 K rows) and an atomic XOR-scatter would
hotspot it, so `lincheck.CscCircuit` now **sorts the nonzeros by column once (host)**
then per fold runs a device **prefix-XOR scan + segment-boundary diff + clean
scatter-set** (`_seg_xor_fold`). Byte-identical to the old host scatter, **491 ms →
0.48 ms (1256×)**. Shared by sha2 (re-gated green).

**Bench (RTX 5090 clmad vs same-instance flock x86 BaseFold, device CSC fold):**

| m | flock CPU (BaseFold, x86 scalar) | GPU flock-zorch | speedup |
|---|---|---|---|
| 22 | 56.5 ms | 64.5 ms | 0.9× |
| 26 | 267 ms | 115 ms | **2.3×** |
| 28 | 1049 ms | 283 ms | **3.7×** |

Crossover ~m=23–24; the win grows with m (the fixed sequential FS round-trips
amortize against the bulk NTT/URM/FRI the GPU crushes). 3.7×@m28 is the honest
**real-circuit BaseFold** number — comparable to sha2's 3.2×@m28, and below the
dense-identity 16× because (a) a real sparse R1CS lets the CPU hit sparse fast-paths
the dense prover skips, and (b) BaseFold is not flock's headline backend (Ligerito
is). Same x86-scalar caveat as elsewhere ([[flock-baseline-needs-macbook]]).

This repo is built **on `zorch`** (sp1-zorch-style bzlmod, `MODULE.bazel`): it
reuses zorch's scheme-agnostic spine (`Sha256Transcript`, sumcheck fold
primitives, `poly.eq`, `pcs.fold`/`basefold`) and keeps only the
byte-identity-critical flock pieces here. GPU byte-match gates run on the zorch
venv (`PYTHONPATH=python:third_party/zorch`), not hermetic Bazel.

### First full sub-protocol: PCS commit, byte-identical + 10×
`pcs::commit` (pack → zero-pad → interleaved forward NTT → SHA-256 Merkle →
32-byte root) reproduces flock's root bit-for-bit, with the dominant encode (NTT)
far past 10×. Verified across **6 configs** (m=18–26, RS rate 1/2 & 1/4, interleave
2–32) by `testing/run_commit_gates.sh` → `artifacts/commit_gate_results.txt`:

| m | rate | batch | root byte-match | encode CPU→GPU | full commit (Merkle on CPU) |
|---|---|---|---|---|---|
| 20 | 1/2 | 32 | ✅ | 20.6× | ~18× |
| 24 | 1/2 | 32 | ✅ | **259.9×** | ~58× |
| 26 | 1/2 | 32 | ✅ | **381.0×** | ~108× |
| 22 | 1/4 | 2 | ✅ | 190.8× | — |
| 22 | 1/2 | 8 | ✅ | 130.0× | — |
| 18 | 1/4 | 16 | ✅ | 11.2× | — |

### Why the GPU wins the *commit phase* (field arithmetic), not SHA-256
flock's own PCS-commit breakdown (`cargo bench -p flock-prover --bench pcs_commit`)
shows the **NTT dominates Merkle by 96–322×** at production sizes (m≥24): e.g.
m=28 → NTT 807 ms vs Merkle 2.5 ms. SHA-256 has dedicated CPU hardware (SHA-NI),
so the GPU does *not* beat it — but Merkle is **<1% of the PCS commit**, so that
doesn't matter *for the commit*. The commit win comes from the field arithmetic
(NTT), which clmad + GPU width crush; the tiny Merkle can stay on the host.

**Scope (honest):** the **full prover is now built, byte-identical, and measured**
— `prover.prove_fast` (commit→bind→zerocheck→lincheck→batched dual-claim open on
one shared challenger, device-resident) reproduces flock `prove`'s `R1csProof`
bit-for-bit (`e2e_oracle_test.py`, m=13–20) and is **9.5× @m=26 / 16.6× @m=28**
vs the same identity-R1CS CPU prover (`bench_e2e_cpu.rs`). The win **grows** with
m — flock's sequential Fiat-Shamir SHA-256 hash chain (the worried-about critical
path) does **not** dominate: the per-round host round-trips are cheap vs the bulk
NTT/URM/FRI work the GPU crushes. Two honest caveats on the headline: (a) the CPU
baseline is **x86 scalar** (flock's NEON paths are aarch64-gated — Apple-silicon
would narrow the gap; a MacBook baseline is the true equivalence test); (b) the
identity R1CS is **dense/degenerate** — a real sparse R1CS lets the CPU hit sparse
fast-paths the generic dense prover skips (vs flock's faster blake3 CPU config the
fused prover is ~2.4× @m=26). The ≥10× holds for the same-instance comparison.

### Sumcheck arithmetic core (`sumcheck.py`)
The reusable kernels shared by **both** sumchecks in flock's PIOP (zerocheck and
lincheck): `build_eq` (eq-table expansion by parallel power-of-two doubling),
`fold_single`/`fold_pair` (bind the low multilinear variable), and `round_pair`
(the Karatsuba ∞-trick round message `(r₀·G(1), G(∞))`). All pure GF(2¹²⁸) over
uint64 lanes → they inherit clmad on GPU and are fully data-parallel. Byte-identical
to flock-core's `zerocheck::{univariate_skip,multilinear}` under **both** the
software mul and the clmad FFI (`sumcheck_oracle_test.py`). GPU-vs-CPU on `build_eq`
(the dominant primitive), vs unmodified x86 flock:

| n | 2ⁿ elems | CPU flock | GPU zorch | speedup |
|---|----------|-----------|-----------|---------|
| 16 | 65 536 | 2.30 ms | 0.116 ms | 19.7× |
| 18 | 262 144 | 9.25 ms | 0.155 ms | 59.9× |
| 20 | 1 048 576 | 37.70 ms | 0.216 ms | **174.4×** |

CPU baseline is flock's x86 scalar build_eq — the only path that compiles on x86;
flock is tuned for Apple silicon (NEON, aarch64-gated), so the definitive
apples-to-apples comparison needs flock built on a MacBook. These primitives are
the building blocks of the full zerocheck/lincheck prove loop (next).

## Performance: GPU vs CPU (the headline gate)
flock-zorch's GPU additive NTT (the dominant PCS-commit primitive) vs
**unmodified succinct flock** on the *same* x86 box — both built at their best
(flock: thin-LTO, `codegen-units=1`, `target-cpu=native`), byte-identical output:

| log_d | CPU flock (scalar) | GPU flock-zorch (clmad) | speedup |
|-------|-------|-------|---------|
| 16 | 13.1 ms | 0.099 ms | **133×** |
| 18 | 58.9 ms | 0.200 ms | **295×** |
| 20 | 290.2 ms | 0.636 ms | **456×** |

The GPU win is the PTX `clmad` carryless-multiply-add (hardware GF(2¹²⁸) mul,
memory-bound). flock's CPU path is its software bit-by-bit clmul — the only one
that compiles on x86 (its NEON/parallel paths are `aarch64+aes`-gated). Reproduce:

```bash
cargo build --release --example bench_ntt_cpu          # CPU anchor (flock-matched flags)
export PATH="$HOME/.local/cuda13/bin:$PATH"            # clmad cubin assembler
JAX_PLATFORMS=cuda PYTHONPATH=python:third_party/zorch "$VENV" python/flock_zorch/testing/cpu_vs_gpu.py
```

```bash
# additive-NTT gate (dumps golden from flock-core's forward_transform_scalar, then
# checks the jax port + self-computed twiddles match byte-for-byte on GPU):
cargo run --release --example dump_ntt -- 12 artifacts/ntt_golden.bin
JAX_PLATFORMS=cuda PYTHONPATH=python:third_party/zorch "$VENV" python/flock_zorch/testing/ntt_oracle_test.py
```

## Toolchain
- GPU: RTX 5090 (sm_120), CUDA 12.9. zorch venv: `/home/jooman/fractalyze/zorch/.venv`
  (jax_fork jax-cuda12 stack `0.10.0.dev*` + `jax-cuda12-pjrt` + `zk_dtypes` 0.0.7).
- Rust: standalone rustup in `~/.cargo` (flock is edition 2024). `flock-core` is a
  path dep at `third_party/flock/crates/flock-core` — the byte-compare baseline.

## Run the field byte-match gate
```bash
# 1. Dump golden (a, b, a*b) triples from flock-core's reference multiply:
cargo run --release --example dump_field_mul -- 1048576 artifacts/field_mul_golden.bin

# 2. Check the jax port reproduces every product byte-for-byte, on GPU:
VENV=/home/jooman/fractalyze/zorch/.venv/bin/python
JAX_PLATFORMS=cuda PYTHONPATH=python:third_party/zorch "$VENV" python/flock_zorch/testing/field_oracle_test.py
# ... and the known-answer vectors:
JAX_PLATFORMS=cuda PYTHONPATH=python:third_party/zorch "$VENV" python/flock_zorch/testing/field_test.py
```
