# clmad — GF(2^128) multiply at the bandwidth ceiling (optimization #1)

Validated GPU optimization of flock's `ghash_mul` using the PTX **`clmad`**
(carryless multiply-add) instruction — the user's original hint. The 256-step
software bit-loop collapses to **8 `clmad` instructions + a shift/XOR reduction**.

> **Status (2026-07): archival optimization record.** flock's software field
> (`field.py` compute) and its `dump_field_mul` golden / `field_oracle_test.py`
> gate were retired in #44/#45 — the in-prover binary-field multiply is now the
> `binary_field_ghash` zk_dtypes dtype (`*`), emitted by the zkx compiler. This
> directory records the `clmad` primitive's validation and stays the reference
> for the intended zkx codegen optimization (below). `validate.py` / `ffi_test.py`
> byte-match against the now-removed `artifacts/field_mul_golden.bin`, so they
> need a regenerated reference to re-run.

## Result (RTX 5090, sm_120)
| impl | throughput | vs ceiling |
|------|-----------|-----------|
| software `fori_loop` (jax, in-prover, pre-#44) | 0.122 G mul/s | 270× below |
| **`clmad` kernel (this)** | **32.75 G mul/s · 1572 GB/s** | **= XOR-add ceiling (memory-bound)** |

**268× faster, byte-identical to flock** (validated against `artifacts/field_mul_golden.bin`,
1,048,576 pairs). The multiply is now fully memory-bandwidth-bound — there is no
compute headroom left; `clmad` makes GF(2^128) multiplication effectively free.

## `clmad` spec (PTX ISA 9.3 §9.7.1.5)
```
clmad.lo.u64 d, a, b, c;   // d = clmul(a,b)[63:0]   ^ c
clmad.hi.u64 d, a, b, c;   // d = clmul(a,b)[127:64] ^ c
```
Carryless 64×64→128 product, `.lo`/`.hi` selects the half, then XOR-adds `c`.
Introduced in PTX 9.3; **requires sm_80+**. The built-in XOR-add folds the
schoolbook cross-terms, so the 128×128→256 unreduced product = exactly 8 `clmad`.

## Toolchain (the gating dependency)
- The instruction needs **PTX ISA 9.3 → CUDA 13.x ptxas**. The system `ptxas 12.9`
  caps at PTX 8.8 and rejects it. A standalone `cuda_nvcc` 13.3.33 is persisted at
  **`~/.local/cuda13/`** (downloaded from NVIDIA's CUDA redistrib;
  `.../redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.3.33-archive.tar.xz`).
- Driver 590.48 (CUDA driver 13.1) **loads the sm_120 SASS cubin** built by ptxas
  13.3 (minor-version compat); it does NOT JIT PTX 9.3, so we assemble to SASS.

## Reproduce
```bash
VENV=.venv/bin/python
~/.local/cuda13/bin/ptxas -arch=sm_120 -O3 optim/clmad/ghash_mul.ptx -o /tmp/ghash.cubin
$VENV optim/clmad/validate.py   # byte-match + bench (needs a regenerated golden — see Status)
```

## Integration status / next
This validates the **primitive** standalone via the CUDA driver API (`cuda_run.py`,
ctypes → libcuda). It is NOT yet wired into the jax-exported prover: the zkx PJRT
plugin uses CUDA 12.x and cannot emit `clmad`. To make the in-prover binary-field
multiply (the `binary_field_ghash` dtype's `*`, and the NTT which is mul-bound)
use `clmad`, EITHER:
1. **zkx compiler path** (the intended "optimize in zkx"): make zkx's GPU codegen
   use ptxas 13.3 and emit `clmad.lo/hi.u64` for binary-field mul (a prime-ir
   `SpecializeBinaryFieldToNVPTX` inline-asm pass + bump the emitted `.version`).
2. **jax FFI custom_call** to this cubin (lighter, but a custom-call boundary per op).
Either way, the heavy hash-circuit `*_oracle_test` gates (keccak / sha2 / blake3,
byte-matched vs flock) are the correctness oracle — `field_oracle_test.py` itself
was retired with the software field.
