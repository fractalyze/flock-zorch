# flock-zorch

GPU port of [succinctlabs/flock](https://github.com/succinctlabs/flock) — an
R1CS-over-GF(2) PIOP prover — onto Fractalyze's zorch / zkx compiler stack,
mirroring `bellman-zorch` / `accumulation-zorch`. See `CLAUDE.md` for the
architecture and non-negotiables.

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
| zerocheck prove (full PIOP) — URM + multilinear rounds | 🚧 in progress (challenger + primitives done) | — |
| PCS open (FRI) / e2e proof | ⏳ | — |
| PCS open (FRI) / e2e proof | ⏳ | — |
| fused `.mlirbc` + Rust host (PJRT) | ⏳ | — |

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

**Scope (honest):** the ≥10× result above is **measured for the PCS commit only**
(a complete, byte-serializable sub-protocol). The full prover (zerocheck +
lincheck sumchecks → PCS open) is **not yet ported/measured** — those are
field-arithmetic-heavy too (so the same clmad win is *expected*), but the
end-to-end speedup must still be measured, and in particular flock's **Fiat-Shamir
SHA-256 is a strictly sequential host hash chain** whose per-round cost could
become the critical path once the bulk sumcheck work moves to the GPU. No
full-prover 10× is claimed here until that is built and benchmarked.

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
JAX_PLATFORMS=cuda PYTHONPATH=python "$VENV" python/flock_zorch/testing/cpu_vs_gpu.py
```

```bash
# additive-NTT gate (dumps golden from flock-core's forward_transform_scalar, then
# checks the jax port + self-computed twiddles match byte-for-byte on GPU):
cargo run --release --example dump_ntt -- 12 artifacts/ntt_golden.bin
JAX_PLATFORMS=cuda PYTHONPATH=python "$VENV" python/flock_zorch/testing/ntt_oracle_test.py
```

## Toolchain
- GPU: RTX 5090 (sm_120), CUDA 12.9. zorch venv: `/home/jooman/fractalyze/zorch/.venv`
  (jax 0.10.0.dev fork + `zk_dtypes` 0.0.7 + zkx CUDA PJRT plugin).
- Rust: standalone rustup in `~/.cargo` (flock is edition 2024). `flock-core` is a
  path dep at `../flock/crates/flock-core` — the byte-compare baseline.

## Run the field byte-match gate
```bash
# 1. Dump golden (a, b, a*b) triples from flock-core's reference multiply:
cargo run --release --example dump_field_mul -- 1048576 artifacts/field_mul_golden.bin

# 2. Check the jax port reproduces every product byte-for-byte, on GPU:
VENV=/home/jooman/fractalyze/zorch/.venv/bin/python
JAX_PLATFORMS=cuda PYTHONPATH=python "$VENV" python/flock_zorch/testing/field_oracle_test.py
# ... and the known-answer vectors:
JAX_PLATFORMS=cuda PYTHONPATH=python "$VENV" python/flock_zorch/testing/field_test.py
```
