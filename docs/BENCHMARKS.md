# flock-zorch Benchmark Results

Apple-to-apple comparison: **unmodified flock CPU** vs **flock-zorch GPU**, same
machine, byte-identical output confirmed before timing.

## Machine Specification

| Component | Detail |
|-----------|--------|
| GPU | 2x NVIDIA GeForce RTX 5090 (sm_120, 32607 MiB each) |
| CPU | AMD Ryzen 9 9950X3D 16-Core (24 threads) |
| RAM | 128 GB DDR5 |
| CUDA Driver | 580.126.09 (CUDA 13.0) |
| ptxas | 13.3 V13.3.33 (for clmad cubin) |
| OS | Linux 6.17.0-35-generic x86_64 |
| Rust | 1.94.0 (stable) |
| Python | 3.11.15 |
| JAX | 0.0.5.dev20260623111059 (fractalyze fork) |
| Date | 2026-06-29 UTC |

### Pinned Commits

```
third_party/flock  @ 73f72028161eaf5cae42c9a054fbfc3f1464fc12 (heads/main)
third_party/zorch  @ 39396626bab059ecd218a7a3dc35e9acdd9b7f42 (flock-byte-fiat-shamir)
```

### Build Configuration

- **clmad**: YES (ptxas 13.3, sm_120 cubin + XLA FFI handler)
- **CPU baselines**: flock built with thin-LTO, `codegen-units=1`, `target-cpu=native`
  (flock's honest best on x86)
- Machine verified idle before benchmarks: 0% GPU util, load avg 1.72/24 cores
- CPU baselines collected while machine was idle; GPU benchmarks ran with GPU idle
  (0% util, 15 MiB)

---

## Byte-Identity Verification (MANDATORY — before timing)

Byte-identity confirmed across all layers before benchmarking:

| Gate | Backend | mul | Result |
|------|---------|-----|--------|
| field-mul (1M pairs) | GPU | clmad | PASS |
| additive NTT (log_d=12) | GPU | clmad | PASS |
| SHA-256 (4096 digests) | GPU | — | PASS |
| Merkle root (4096 leaves) | GPU | — | PASS |
| merkle_multi proof | GPU | — | PASS |
| PCS commit root (m=20) | GPU | software | PASS |
| sumcheck (build_eq/round_pair/fold) | GPU | clmad | PASS |
| challenger (SHA-256 duplex) | GPU | — | PASS |
| gf8 URM (phi8 + round1) | GPU | — | PASS |
| FRI fold | GPU | software | PASS |
| row_batch fold | GPU | software | PASS |
| **e2e full prover (m=13, 39 checks)** | **GPU** | **clmad** | **PASS** |
| **Keccak BaseFold (m=19)** | **GPU** | **clmad** | **PASS** |
| **Keccak3 Ligerito (m=22)** | **GPU** | **clmad** | **PASS** |

---

## Results: Per-Layer Micro-Benchmarks (clmad)

### Additive NTT — `cpu_vs_gpu.py` + `bench_ntt_cpu`

| log_d | N elements | flock CPU (ms) | GPU clmad (ms) | speedup |
|-------|-----------|---------------|---------------|---------|
| 16 | 65,536 | 12.76 | 0.159 | **80x** |
| 18 | 262,144 | 54.27 | 0.329 | **165x** |
| 20 | 1,048,576 | 252.34 | 0.709 | **356x** |
| 22 | 4,194,304 | 1,207.0 | 2.715 | **445x** |
| 24 | 16,777,216 | 5,107.6 | 18.659 | **274x** |

Byte-identity at log_d=12: PASS. GATE PASS (>=10x at all sizes). Peak 445x at
log_d=22. The clmad kernel is memory-bandwidth-bound (1558 GB/s = XOR-add ceiling).

### Sumcheck build_eq — `sumcheck_gpu_vs_cpu.py` + `bench_sumcheck_cpu`

| n | 2^n elements | flock CPU (ms) | GPU clmad (ms) | speedup |
|---|-------------|---------------|---------------|---------|
| 16 | 65,536 | 2.69 | 0.151 | **18x** |
| 18 | 262,144 | 10.72 | 0.198 | **54x** |
| 20 | 1,048,576 | 40.12 | 0.271 | **148x** |
| 22 | 4,194,304 | 191.80 | 0.541 | **355x** |
| 24 | 16,777,216 | 770.68 | 2.071 | **372x** |

Byte-identity: PASS. GATE PASS (>=10x at all sizes).

### Sumcheck Core (full breakdown) — `bench_all.py` (clmad)

| log_n | build_eq (ms) | round_pair (ms) | fold (ms) | build_eq (G elem/s) |
|-------|--------------|----------------|----------|-------------------|
| 16 | 0.161 | 0.207 | 0.052 | 0.41 |
| 18 | 0.178 | 0.218 | 0.053 | 1.48 |
| 20 | 0.231 | 0.263 | 0.053 | 4.54 |

### XOR-add Bandwidth Ceiling

2^24 elements: **32.5 G add/s, 1558 GB/s** — confirms the RTX 5090's memory
bandwidth is saturated; clmad makes GF(2^128) mul effectively free (268x faster
than software fori_loop).

---

## Results: Full-Prover Benchmarks (clmad)

### Identity R1CS e2e — `e2e_fused_bench.py` + `bench_e2e_cpu`

| m | flock CPU (ms) | GPU clmad (ms) | speedup |
|---|---------------|---------------|---------|
| 22 | 72.19 | 63.19 | **1.1x** |
| 26 | 1,207.24 | 111.20 | **10.9x** |
| 28 | 4,742.82 | 308.85 | **15.4x** |

Byte-identity: PASS (e2e_oracle_test, m=13, all 39 fields). GPU wins grow with m
because the fixed sequential Fiat-Shamir round-trips amortize against the bulk
NTT/URM/FRI the GPU crushes. At m=22 the overhead dominates; at m=28 the GPU is
15.4x faster.

### Keccak3 Ligerito (real hash circuit) — `e2e_keccak3_ligerito_bench.py`

| Workload | m | flock CPU (ms) | GPU clmad (ms) | speedup |
|----------|---|---------------|---------------|---------|
| Keccak3 Ligerito | 22 | 46.01 | 478.41 | 0.10x |

GPU is 10x slower at m=22. The Keccak3 Ligerito prover at this size has many
sequential Fiat-Shamir rounds and the per-round data-parallel work is too small
to amortize GPU launch overhead. The GPU would need larger m to win (matching the
identity R1CS pattern: GPU breaks even ~m=22, wins at m>=26).

### Per-Phase GPU Breakdown — `prover_phase_gpu_bench.py` (clmad)

| m | commit (ms) | zerocheck (ms) |
|---|------------|---------------|
| 20 | 104.6 | 26.6 |
| 22 | 123.0 | 32.2 |
| 24 | 139.3 | 38.2 |

Commit (pack + interleaved NTT + Merkle) dominates at ~78-80% of GPU time.

### SHA-256 / BLAKE3 (NOT COLLECTED)

The SHA-256 and BLAKE3 provers (BaseFold and Ligerito) hit a JAX limitation in the
CSC lincheck fold (`_seg_xor_fold`):

```
NotImplementedError: non-zero interior padding is not supported: (1, 0)
```

This is unrelated to clmad — it's a JAX 0.0.5.dev limitation with the padding
operation used by `lincheck.CscCircuit`. The Keccak provers (which use a different
lincheck circuit) are unaffected.

**CPU baselines** (flock's honest best on x86, collected while machine was idle):

| Workload | Config | flock CPU (ms) |
|----------|--------|---------------|
| SHA-256 BaseFold | n_comp=8, m=18 | 10.20 |
| SHA-256 Ligerito | n_comp=128, m=22 | 44.82 |
| BLAKE3 BaseFold | n_comp=8, m=17 | 47.22 |
| Keccak3 Ligerito | n_keccaks=49, m=22 | 46.01 |

---

## Software field.mul Comparison (no clmad)

For completeness, these numbers show the GPU WITHOUT clmad (software field.mul).
The software field.mul (64-step fori_loop per GF(2^128) product) is the bottleneck.

### NTT (software field.mul)

| log_d | flock CPU (ms) | GPU software (ms) | speedup |
|-------|---------------|-------------------|---------|
| 16 | 14.38 | 48.47 | 0.3x |
| 18 | 59.94 | 55.70 | 1.0x |
| 20 | 275.74 | 108.44 | 2.5x |
| 22 | 1,207.0 | 216.31 | 5.6x |
| 24 | 5,107.6 | 1,687.9 | 3.0x |

### Sumcheck build_eq (software field.mul)

| n | flock CPU (ms) | GPU software (ms) | speedup |
|---|---------------|-------------------|---------|
| 16 | 4.23 | 98.47 | 0.04x |
| 18 | 10.50 | 119.42 | 0.09x |
| 20 | 47.23 | 138.13 | 0.34x |
| 22 | 191.80 | 161.94 | 1.2x |
| 24 | 770.68 | 360.14 | 2.1x |

Without clmad, the full-prover GPU benchmarks cannot run due to an XLA
argument-packing limit in `_batch_inv` (2033 device memory arguments > max 1024).
The 127-step Fermat-inverse loop with software field.mul unrolls to too many XLA
operations; clmad traces as a single FFI op per multiply and stays under the limit.

---

## Methodology

- **Warm-up**: 1 warm-up call (JIT compile + first kernel launch excluded)
- **Timing**: best-of-3 (full prover) or best-of-5/10/50 (micro-benchmarks),
  consistent with upstream flock methodology (BENCHMARKS.md §0)
- **CPU threading**: single-threaded (matches flock's default bench; flock's
  parallel/NEON paths are aarch64-gated)
- **GPU device**: cuda:0 (first RTX 5090); no multi-GPU
- **Environment**:
  ```bash
  export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
  export PYTHONPATH=python:third_party/zorch
  export FLOCK_CLMAD_CUBIN=$(pwd)/optim/clmad/ghash_mul.cubin
  ```

---

## Caveats (HONEST)

1. **CPU baseline is x86 SCALAR.** flock's NEON paths are aarch64-gated, so Apple
   silicon would narrow the gap. The definitive equivalence test wants flock built
   on a MacBook.

2. **Identity R1CS is dense/degenerate.** A real sparse R1CS (SHA-256, BLAKE3,
   Keccak) lets the CPU hit fast-paths the generic dense prover skips. Report BOTH
   identity-e2e AND real-hash-circuit numbers: the Keccak3 Ligerito at m=22 shows
   the GPU is 10x slower (sequential FS rounds dominate at this size).

3. **SHA-256 / BLAKE3 GPU numbers not collected.** The CSC lincheck fold hits a
   JAX interior-padding `NotImplementedError` in this jaxlib version. Keccak
   provers (different lincheck circuit) are unaffected.

4. **GPU wins scale with m.** At m=22 the GPU barely breaks even on identity R1CS
   and is 10x slower on the Keccak3 real circuit. The GPU advantage requires large
   witness sizes (m>=26) where the bulk data-parallel work amortizes the sequential
   Fiat-Shamir round-trip overhead.

---

## Commands Used

```bash
# Setup
~/.local/cuda13/bin/ptxas -arch=sm_120 -O3 optim/clmad/ghash_mul.ptx \
    -o optim/clmad/ghash_mul.cubin
# (build libghash_clmad.so with XLA FFI headers, API version 0.1 for zkx compat)

# CPU baselines (collected on idle machine, load avg 1.72/24 cores)
target/release/examples/bench_ntt_cpu <log_d> <iters>
target/release/examples/bench_sumcheck_cpu <n> <iters>
target/release/examples/bench_e2e_cpu <m> <iters>
target/release/examples/bench_sha2_cpu
target/release/examples/bench_sha2_ligerito_cpu
target/release/examples/bench_blake3_cpu
target/release/examples/bench_keccak3_ligerito_cpu

# GPU benchmarks (clmad)
$VENV python/flock_zorch/testing/cpu_vs_gpu.py
$VENV python/flock_zorch/testing/sumcheck_gpu_vs_cpu.py
$VENV python/flock_zorch/testing/bench_all.py
$VENV python/flock_zorch/testing/e2e_fused_bench.py 22 26 28
$VENV python/flock_zorch/testing/e2e_keccak3_ligerito_bench.py
$VENV python/flock_zorch/testing/prover_phase_gpu_bench.py

# GPU benchmarks (software field.mul, for comparison)
# same commands without FLOCK_CLMAD_CUBIN

# Byte-identity (all layers + full prover, GPU with clmad)
$VENV python/flock_zorch/testing/e2e_oracle_test.py
$VENV python/flock_zorch/testing/keccak_oracle_test.py
$VENV python/flock_zorch/testing/keccak3_ligerito_oracle_test.py
```
