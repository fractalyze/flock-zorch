# flock-zorch Benchmark Results

Apple-to-apple comparison: **unmodified flock CPU** vs **flock-zorch GPU**, same
machine, byte-identical output confirmed before timing. Every instance is a **real
flock hash-circuit R1CS** (SHA-256, BLAKE3, Keccak3) at the size flock itself
ships — swept over the witness size `m` to locate the GPU/CPU crossover. No
synthetic circuits.

## Machine Specification

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA GeForce RTX 5090 (sm_120, 32607 MiB) |
| CPU | AMD Ryzen 9 9950X3D 16-Core |
| CUDA | driver 13.x; frx runtime = CUDA 12 (frx-cuda12 stack); ptxas ≥ 13.3 for compiler-emitted clmad (sm_120) |
| Rust | flock built thin-LTO, `codegen-units=1`, `target-cpu=native` (its honest x86 best) |
| Python / frx | 3.11 / frx jax-fork cuda12 stack |
| Date | 2026-06-30 UTC (BaseFold); Keccak3 Ligerito re-measured 2026-07-16 UTC |

- **clmad**: YES (compiler-emitted; ptxas 13.3 on PATH) — hardware GF(2¹²⁸) multiply.
- **Idle**: GPU verified idle (0% util, 0 compute procs) before every GPU timing;
  CPU baselines collected on an idle machine.

---

## Method

- **Full prover** `prover.prove_fast` → `R1csProof` (commit → bind → zerocheck →
  lincheck → batched dual-claim open, one shared challenger, device-resident),
  reproducing flock `prove`'s proof **bit-for-bit**.
- **Same instance both sides.** The Rust CPU baseline and the JAX GPU prover run
  the *same* R1CS: the golden is dumped from flock-core (`dump_sha2` /
  `dump_blake3` / `dump_keccak3_ligerito`), the CPU bench proves it
  (`bench_*_cpu`), and the GPU bench ingests the same golden.
- **m = K_LOG + n_blocks_log.** SHA-256 `K_LOG=15`, Keccak3 `K_LOG=17`. m is swept
  by the instance count (SHA-256 `n_comp`, Keccak3 `n_keccaks`).
- **Timing**: best-of-3 (full prover) after one warm-up (JIT compile excluded).
  CPU single-threaded (flock's parallel/NEON paths are aarch64-gated).
- **Env**: `JAX_PLATFORMS=cuda`, `PATH` includes `~/.local/cuda13/bin` (ptxas 13.3 → clmad),
  `PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)"` (zorch via the git_override).

---

## Byte-Identity Verification (MANDATORY — before timing)

Confirmed bit-for-bit on GPU (fixtures dumped from unmodified flock-core):

| Gate | Backend | Result |
|------|---------|--------|
| field-mul / additive NTT / SHA-256 / Merkle root | clmad | PASS |
| sumcheck (build_eq / round_pair / fold) / challenger | clmad | PASS |
| zerocheck / lincheck / ring-switch / BaseFold / `pcs::open` | clmad + sw | PASS |
| **e2e full prover** `prove_fast` → `R1csProof` | clmad | PASS |
| **SHA-256 / BLAKE3 / Keccak-f[1600] / Keccak3** (BaseFold + Ligerito) | clmad | PASS |

---

## Results: Full Prover on Real Hash Circuits (clmad)

### SHA-256 (BaseFold) — crossover ≈ m=24

| m  | n_comp | flock CPU (ms) | GPU clmad (ms) | speedup |
|----|--------|----------------|----------------|---------|
| 18 | 8      | 11.5           | 51.9           | 0.22×   |
| 20 | 32     | 13.8           | 51.0           | 0.27×   |
| 22 | 128    | 24.3           | 59.8           | 0.41×   |
| 24 | 512    | 67.0           | 66.6           | **1.01×** |
| 26 | 2048   | 256.8          | 94.5           | **2.7×** |

### BLAKE3 (BaseFold)

| m  | n_comp | flock CPU (ms) | GPU clmad (ms) | speedup |
|----|--------|----------------|----------------|---------|
| 26 | 4096   | 280.3          | 96.0           | **2.9×** |

### Keccak3 (Ligerito) — crossover ≈ m=24  *(re-measured 2026-07-16)*

| m  | n_keccaks | flock CPU (ms) | GPU clmad (ms) | speedup |
|----|-----------|----------------|----------------|---------|
| 22 | 49        | 27.0           | 47.2           | 0.57×   |
| 24 | 384       | 71.3           | 66.4           | **1.07×** |
| 26 | 1536      | 277.1          | 94.1           | **2.9×** |
| 28 | 6144      | 1,167.1        | 185.7          | **6.3×** |

Re-measured on the device-native Ligerito open now on `main` (zorch #479 — the
recursive open compiles to one device program — plus on-device query-position
sampling, #104): GPU prove ~3× faster than the 2026-06-30 baseline, crossover moved
**left from ≈27 to ≈24** (GPU wins from m=24), **6.3×** by m=28, which now fits in
32 GB with no allocation retry.

<sub>Prior baseline (2026-06-30, host-sampled queries, pre-#479): 254.3 / 277.7 /
401.3 / 627.0 ms GPU at m=22/24/26/28, crossover ≈27, m=28 memory-bound (16 GB retry).</sub>

---

## Device SHA-256 Fiat-Shamir transcript — measured crossover impact (#7)

The milestone thesis is that moving Fiat-Shamir **on-device** should push the
crossover **left** (kill the per-round host↔device latency). #7 tested the one
drop-in that keeps byte-identity: injecting zorch's device `Sha256` byte hash
(the `zorch.sha256` marker) as the `byte_hash` on `prove_fast`. **Result: it moves
the crossover right, not left** — the byte transcript is a large regression.

Why: the byte device transcript keeps flock's growing `bytes` buffer and re-hashes
it via the `zorch.sha256` marker **per squeeze** — one GPU dispatch + host↔device
transfer for each of a prove's hundreds of `sample`/`grind` calls, each re-hashing
the whole buffer. And the flock `Challenger` still serializes F128 challenges
through host numpy, so the per-round sync it was meant to remove is still there.

`prove_fast`, identity R1CS, RTX 5090, host- vs device-transcript on the **same**
witness/arithmetic (best-of-3, warm). Because only the transcript differs,
`device − host ≈ the pure Fiat-Shamir overhead` (field-mul-independent):

| m  | host-transcript (ms) | device-transcript (ms) | ratio | device − host (FS overhead) |
|----|----------------------|------------------------|-------|-----------------------------|
| 22 | 5,189                | 50,921                 | 9.8×  | **~45.7 s** |
| 24 | 11,907               | 55,406                 | 4.7×  | **~43.5 s** |

Measured with **software** GF(2¹²⁸) mul (no CUDA-13.x `ptxas` on this box), which
inflates the *baseline* prover ~100× but **not** the transcript (its ops are SHA
dispatches, not field muls). So the honest, config-independent figure is the
**~44 s of fixed FS overhead per prove**. Against the real clmad prover (59.8 ms
@ m=22, 66.6 ms @ m=24) that overhead is ~700× the whole prove: the device-byte
GPU prover would be ~44 s versus flock CPU's 24–67 ms — the crossover leaves the
chart to the right at every m here.

**Takeaway.** The marker byte transcript is a correct drop-in — zorch guarantees
it is byte-identical to the host hashlib
(`byte_transcript_test.test_device_substrate_matches_host`), so flock keeps no
device gate of its own — but it is not a perf lever; the `byte_hash` knob stays an
opt-in seam. The left-shift requires the *other* zorch surface — `Sha256FieldTranscript` (fixed-shape
streaming `Sha256State`, `lax.scan`-threadable) — threaded through a **device
sumcheck driver** so challenges stay on-device and the whole round loop
single-dispatches. That is P2 #9 (`sumcheck → zorch device driver`), not a
transcript-backend swap.

---

## Interpretation

flock's prover is a **sequential SHA-256 Fiat-Shamir chain**: each round samples a
challenge from the transcript before the next round's bulk work can start. At small
m the per-round data-parallel work (NTT butterflies, the round-1 URM, FRI folds) is
too small to amortize GPU kernel-launch + host↔device latency, so the CPU wins. As
m grows the bulk work per round grows and the GPU overtakes:

- **BaseFold circuits (SHA-256, BLAKE3): crossover ≈ m=24.** GPU 2.7–2.9× by m=26.
- **Ligerito (Keccak3): crossover ≈ m=24** (device-native open, was ≈27). Compiling
  the recursive open to one device program (#479) + on-device query sampling removed
  the mid-recursion host syncs; GPU **6.3×** by m=28.

Above the crossover the advantage grows with m — consistent with the GPU winning on
bulk arithmetic and the CPU winning on latency-bound small instances.

---

## Caveats (HONEST)

1. **CPU baseline is x86 SCALAR.** flock's NEON paths are aarch64-gated, so Apple
   silicon would shift the crossover to the right. The definitive equivalence test
   wants flock built on a MacBook.
2. **The GPU only wins above the crossover** (m ≳ 24 BaseFold, ≳ 24 Ligerito);
   below it the CPU is faster. The large-m numbers are not a universal speedup.
3. **zorch is early-stage.** This codebase keeps the CPU prover readable while
   enabling GPU codegen; field-arithmetic optimization is delegated to the ZKX/MLIR
   compiler, which is still maturing — that is why the crossover sits at large m.
4. **Keccak3 m=28** now fits in 32 GB with no allocation retry (post #479); the older
   pre-#479 baseline hit a 16 GB retry and was memory-bound there.

---

## Commands Used

```bash
# clmad fast path: put a CUDA 13.3 ptxas on PATH (sm_120 requires >= 13.3); the
# pinned frx wheel's compiler then emits hardware clmad. Nothing to build.
export PATH="$HOME/.local/cuda13/bin:$PATH"

VENV=.venv/bin/python                          # built by scripts/setup.sh
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)"   # zorch via the git_override

# one sweep point (SHA-256, m via n_comp): dump real R1CS -> CPU anchor -> GPU
cargo run --release --example dump_sha2 -- <n_comp> artifacts/sha2_golden.bin
target/release/examples/bench_sha2_cpu <n_comp>                       # flock CPU ms
$VENV python/flock_zorch/testing/e2e_sha2_bench.py <cpu_ms>           # GPU vs CPU

# Keccak3 Ligerito (m via n_keccaks)
cargo run --release --example dump_keccak3_ligerito -- <n_keccaks> artifacts/keccak3_ligerito_golden.bin
target/release/examples/bench_keccak3_ligerito_cpu <n_keccaks>
$VENV python/flock_zorch/testing/e2e_keccak3_ligerito_bench.py <cpu_ms>

# BLAKE3
cargo run --release --example dump_blake3 -- 4096 artifacts/blake3_golden.bin
target/release/examples/bench_blake3_cpu 4096
$VENV python/flock_zorch/testing/e2e_blake3_bench.py <cpu_ms>
```
