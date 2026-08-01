# GF(2^128) primitive census — 2026-08-01

This is the checked-in result of the first census run, not a vendor-spec
projection.  The card was idle (RTX 5090, 32 GiB), the CPU comparison ran on
the same machine, and the GPU numbers use a locally built PJRT from
`fractalyze/xla@28e75c6243`.  The pinned flock oracle is
`succinctlabs/flock@73f72028`.

The repository's published `frx==0.10.0.dev20260725093147` wheel was checked
separately with the same CUDA 13.3 assembler.  It also emitted eight `clmad`
instructions, reproduced 0.031 ns/element multiply and 0.089 ns/element extend,
and completed the same proof in 30.4 ms.  The finding therefore does not depend
on an unreleased XLA build and does not require a dependency bump.

## The result

The large additive-NTT regression was a poisoned baseline: the process was
using CUDA 12.9 `ptxas`, so every GHASH multiply took XLA's portable software
path.  With `CUDA_ROOT=/usr/local/cuda-13.3` and CUDA 13.3.73 first on `PATH`,
the emitted PTX is version 9.3 and contains eight `clmad` instructions per
general GHASH multiply.

At 2^24 field elements:

| operation | software GHASH | hardware `clmad` | speedup |
|---|---:|---:|---:|
| multiply | 1.722 ms | 0.517 ms | 3.33x |
| zerocheck INTT64 -> coset NTT64 extend | 116.226 ms | 1.491 ms | 77.96x |

The fixed-64 extend is therefore neither an 11 ms bandwidth problem nor an NTT
indexing problem on this compiler.  Its multiply-heavy software lowering was
the loss.  This is also why checking only that a CUDA 13 installation exists is
insufficient: the exact assembler selected by PJRT determines the lowering.

## Complete large-size table

Each row below is the median of five timed executions after two warmups at
2^24 elements.  The sweep covered every power from 2^10 through 2^24.  `lane`
is native-field time divided by the byte-identical `uint64[lo,hi]` operation;
a value near one excludes a field-dtype lowering defect.  Roofline percentages
use a measured/vendor DRAM ceiling of 1,790 GB/s.  Compute percentages are
deliberately omitted until a `clmad` issue microbenchmark is available.

| primitive | ns/element | bandwidth roofline | native/lane | finding |
|---|---:|---:|---:|---|
| add | 0.031 | 86.6% | 0.99x | dispatch at small sizes; healthy steady state |
| sub | 0.031 | 86.3% | 0.99x | dispatch at small sizes; healthy steady state |
| multiply | 0.031 | — | — | hardware `clmad`; dispatch at small sizes |
| additive NTT | 0.174 | — | — | dispatch curve |
| INTT64 -> coset NTT64 extend | 0.090 | — | — | dispatch curve after `clmad` |
| XOR-sum reduce | 0.010 | 87.1% | 1.01x | prior 170x dtype loss is absent |
| select | 0.027 | 99.9% | 0.86x | prior 170x dtype loss is absent |
| gather | 0.068 | 29.5% | 1.01x | indexed-load cost, not field lowering |
| scatter | 0.230 | 8.7% | 1.00x | indexed-store cost, not field lowering |
| batch inverse | 6.315 | — | — | multiply/dispatch chain |
| ring-switch bit slices | 0.250 | — | — | dispatch curve |
| ring-switch transpose | 0.171 | — | — | dispatch curve |
| to GHASH bitcast | 0.022 | — | — | alias internally; output-boundary copy |
| from GHASH bitcast | 0.021 | — | — | alias internally; output-boundary copy |

No measured field opcode loses to its lane equivalent.  Consequently the
ranked lowering finding is:

1. Software GHASH multiply, when CUDA 13.3 `ptxas` is not selected.  On the
   complete m=22 proof it costs 68.4 ms: 48.61 ms in zerocheck, 14.01 ms in
   lincheck, 4.50 ms in open, and 1.15 ms in commit.  It also adds 114.7 ms to
   one 2^24-element fixed-64 extend and dominates every other measured gap.
2. No remaining field-dtype lowering defect.  Gather and scatter have the
   lowest bandwidth efficiencies, but their 1.01x/1.00x lane ratios exclude
   the GHASH dtype as the cause.
3. The remaining size curves collapse toward their steady-state cost.  The
   next lever is batching/fusion of the small kernels, not another arithmetic
   rewrite in the prover.

## Same-box proof and correctness gate

For the 256-compression, m=22 BLAKE3 Ligerito fixture, unmodified flock CPU
proved in 63.05 ms.  The same GPU proof with software GHASH took 98.6 ms (2,597
compressions/s), reproducing the anomaly: the GPU was 1.56x slower than CPU.
With hardware `clmad`, flock-zorch proved in 30.2 ms (8,484 compressions/s), so
the GPU is instead 2.09x faster on the same configuration and 3.27x faster than
its poisoned software-GHASH baseline.

| phase | software GHASH | hardware `clmad` | saved/proof |
|---|---:|---:|---:|
| commit | 4.42 ms | 3.27 ms | 1.15 ms |
| zerocheck | 50.63 ms | 2.02 ms | 48.61 ms |
| lincheck | 15.69 ms | 1.68 ms | 14.01 ms |
| open | 27.36 ms | 22.86 ms | 4.50 ms |
| wall | 98.6 ms | 30.2 ms | 68.4 ms |

The full `blake3_ligerito_oracle_test.py` proof comparison passed every
serialized field against the flock fixture.

## Profiler limitation

Nsight Compute 2026.2.1 is installed, but this host rejects hardware-counter
collection with `ERR_NVGPUCTRPERM`; the current user has no passwordless sudo.
No counter-derived claim is made here.  Once profiling is enabled, scatter and
the small-kernel launch sequence are the first targets; absolute `ncu` timings
must not be used because replay serializes kernels.
