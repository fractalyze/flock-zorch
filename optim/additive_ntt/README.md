# additive-NTT — compiler vs binius-gpu (P4 benchmark, issue #23)

Head-to-head between the compiler's fused additive-NTT (`lax.ntt` on binary-field
dtypes → `ntt_pass_fusion` custom kernels) and binius-gpu's hand-written
`additive_ntt_kernel`, to decide whether a hand-written FFI kernel (#22) is worth
building. **It is not** — at equal arithmetic the compiler is ~2× *faster* than the
vendor kernel, and the transform is field-multiply-bound, so the real lever is a
faster multiply in the compiler, not a hand port.

Reference reporting style: [`optim/clmad/README.md`](../clmad/README.md).

## Result (RTX 5090, sm_120, rate 0, kernel-only device time)

### GF(2³²) — head-to-head (both use the Fan-Paar tower software multiply)

Our `lax.ntt` on `binary_field_t5` is **byte-identical** to binius-gpu's
`additive_ntt_kernel<uint32_t, FanPaarTowerField<5>>` — verified against binius's
own pinned MD5 oracle (`additive_ntt_hashes[0][log_h]`), 10/10 (`log_d` 1–10).
On equal arithmetic the compiler wins at every size:

| log_d | binius kernel | compiler `lax.ntt` | compiler faster |
|------:|-------------:|-------------------:|:---------------:|
| 2¹²   | 0.157 ms | 0.047 ms | 3.3× |
| 2¹⁴   | 0.186 ms | 0.043 ms | 4.3× |
| 2¹⁶   | 0.215 ms | 0.050 ms | 4.3× |
| 2¹⁸   | 0.284 ms | 0.165 ms | 1.7× |
| 2²⁰   | 1.160 ms | 0.573 ms | 2.0× |
| 2²²   | 4.169 ms | 2.170 ms | 1.9× |
| 2²³   | 8.402 ms | 4.568 ms | 1.8× |

Plateau throughput: binius ~1.0 Gelem/s, **compiler ~1.9 Gelem/s**.

### GF(2¹²⁸) — compiler only (flock's actual field; binius has no 128-bit kernel)

`binary_field_ghash`, `lax.ntt`:

| log_d | ntt_ms | Gelem/s |
|------:|-------:|--------:|
| 2¹⁶   | 0.075 | 0.879 |
| 2¹⁸   | 0.318 | 0.825 |
| 2²⁰   | 1.152 | 0.910 |
| 2²²   | 5.048 | 0.831 |
| 2²³   | 10.443 | 0.803 |

(`binary_field_t7`, the 128-bit tower, is a further ~4× slower — its tower
multiply is heavier than GHASH's — and plateaus the same way.)

## The transform is compute-bound, not bandwidth-bound

Effective bandwidth at the plateau (elem × itemsize × 2 r/w × ~3 fused passes ÷ time):

- GF(2³²) 2²³: ~44 GB/s — **~3%** of the ~1572 GB/s XOR-add ceiling
- GF(2¹²⁸) 2²³: ~77 GB/s — **~5%** of the ceiling

So the additive-NTT is **field-multiply-bound** (compute/latency), not
memory-bandwidth-bound — matching clmad's note that "the NTT is field-mul-bound".
The multiply that runs today is **software carryless** multiply: prime-ir's
`mlir::prime_ir::field::emitClmul64` emits a bit-manipulation `xor`/`and`/`shl`/`shr`
sequence (confirmed in the emitted PTX — no hardware `clmul`/`clmad` instruction;
the `xla_cuda12` plugin cannot assemble PTX ISA 9.3 `clmad`).

## Verdict for the P4 sub-issues

- **#22 (hand-written software-GHASH FFI kernel) → closed.** A hand-written kernel
  with the *same* software multiply the compiler already uses loses to the compiler
  by ~2× (measured at GF(2³²), where both run the identical Fan-Paar tower mul). The
  compiler schedules the butterfly network + multiplies better than a hand port, so a
  hand-written FFI kernel is the wrong vehicle.
- **#24 (clmad fast path) → kept, re-scoped.** Because the transform is mul-bound,
  the real lever is a faster multiply. But since the compiler out-schedules
  hand-written kernels at equal arithmetic, clmad belongs **in the compiler**: make
  prime-ir's `emitClmul64` emit the hardware `clmad` instruction (clmad README's
  "Option 1") instead of the software sequence. That accelerates *all* binary-field
  multiplies, not just the NTT.
- **#25 (CUDA-13.3 ptxas) → kept.** The blocker for emitting `clmad` at all, and
  independently needed to build/validate the `optim/clmad` primitive on this machine.

## Reproduce

Compiler side + byte-match gate (from the repo root):

```bash
VENV=.venv/bin/python                              # built by scripts/setup.sh
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  $VENV -m flock_zorch.testing.additive_ntt_bench
```

binius-gpu side (kernel-only timing) — the vendor kernel has no benchmark of its
own, so [`binius_kernel_timing.patch`](binius_kernel_timing.patch) adds a
CUDA-event bracket around just the kernel-launch loop (H2D/D2H excluded) plus a
hidden `[bench]` sweep:

Needs a CUDA ≥ 12.8 `nvcc` on `PATH` (sm_120 / RTX 5090 support landed in 12.8):

```bash
cd <binius-gpu checkout>            # github.com/IrreducibleOSS/binius-gpu
git submodule update --init --recursive           # Catch2 + nvbench
git apply <this dir>/binius_kernel_timing.patch
cmake -B build -DCMAKE_CUDA_COMPILER=nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=g++ -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ntt_tests -j"$(nproc)"
./build/ntt_tests "[bench]"          # CSV: log_h,kernel_ms,elem_per_s
```

Both measure device-compute only (input already resident) — the fair comparison,
since in the prover the NTT input is already on device.
