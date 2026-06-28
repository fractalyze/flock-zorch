# Running flock-zorch on a GPU box

flock-zorch is the GPU port; it depends on two pinned sibling repos, vendored as git
submodules under `third_party/`:

| submodule | repo | pin | why |
|---|---|---|---|
| `third_party/flock` | `succinctlabs/flock` (public) | `main` @ `73f7202` | the **byte-compare oracle**: Cargo path-deps `flock-core`/`flock-prover`; `examples/dump_*.rs` dump the golden fixtures from it |
| `third_party/zorch` | `fractalyze/zorch` (private) | `flock-byte-fiat-shamir` @ `39396626` | the **scheme-agnostic spine**: `zorch.hash.sha256`, `zorch.byte_transcript` (FS duplex), `zorch.sumcheck.field_ops` |

> The zorch pin is the `flock-byte-fiat-shamir` branch, not `main` — it carries the
> byte-SHA256 transcript + the `FieldOps` seam flock-zorch reuses. Keep the
> `requirements.in` wheel set in lockstep with this pin (the zkx binary-field GPU
> contract is a hard cut; CPU-only CI can't catch a desync). `MODULE.bazel` points
> bazel at the same submodule via `local_path_override`, so the gate path
> (`PYTHONPATH`) and the bazel path use the identical zorch.

## Prerequisites

- **GPU**: CUDA, sm_120 reference is an RTX 5090 (CUDA 12.9 driver stack). Any
  recent NVIDIA GPU works for the gates; the headline benches assume the 5090.
- **Rust** via rustup (`flock-core` is edition 2024).
- **Python 3.11**.
- **SSH access to `fractalyze/zorch`** (the zorch submodule clones over SSH).
- Optional, for the GPU fast path: a **CUDA 13.x `ptxas`** at `~/.local/cuda13/bin`
  (assembles the `clmad` cubin). Without it everything still runs on the software
  `field.mul` — byte-identical, just slower. See `optim/clmad/README.md`.

## Quick start

```bash
git clone --recursive git@github.com:fractalyze/flock-zorch.git
cd flock-zorch
scripts/setup.sh          # idempotent; ~minutes (venv + cargo build + goldens + smoke gates)
```

If you cloned without `--recursive`, `scripts/setup.sh` runs `git submodule update
--init --recursive` for you.

`scripts/setup.sh` does, in order: preflight → submodules → Python venv (`.venv`,
from `requirements.in` + the fractalyze PyPI index) → `cargo build --release`
(cdylib + dumpers + CPU benches) → optional clmad → regenerate the core goldens →
two **smoke gates** (`field`, `e2e`) that prove byte-identity end-to-end.

## Running the gates

The GPU byte-match gates run on the venv via `PYTHONPATH`, **not** hermetic bazel
(the jax/zkx CUDA wheels + the clmad cubin are not hermetic). Set the environment
once:

```bash
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # don't grab ~75% of VRAM up front (shared-box friendly)
export PYTHONPATH=python:third_party/zorch
VENV=.venv/bin/python                        # or `source .venv/bin/activate`
```

Then run any gate (`jax_enable_x64` is set by the tests themselves):

```bash
$VENV python/flock_zorch/testing/e2e_oracle_test.py          # full prover, identity R1CS
$VENV python/flock_zorch/testing/keccak_oracle_test.py       # Keccak full prove (BaseFold)
$VENV python/flock_zorch/testing/blake3_ligerito_oracle_test.py
# ... one *_oracle_test.py per layer / hash circuit
```

Each gate needs its golden in `artifacts/` (see below). The full per-layer +
per-hash-circuit gate list is the `*_oracle_test.py` set under
`python/flock_zorch/testing/`.

## Golden fixtures

`artifacts/` is gitignored (regenerable, and `blake3_golden.bin` alone is ~118 MB).
Regenerate from the pinned flock:

```bash
scripts/dump_goldens.sh          # core: every layer + identity e2e (fast)
scripts/dump_goldens.sh all      # + the real hash circuits (keccak/sha2/blake3; slow)
```

Two gates sweep configs with their own runners instead of a single default golden:
- **PCS commit** — `python/flock_zorch/testing/run_commit_gates.sh` (6 `(m,rate,batch)` configs)
- **lincheck** — the multi-`(m,k_log,k_skip)` gate regenerates its per-config goldens.

## Optional: clmad GPU acceleration

The carryless `field.mul` fast path is the PTX `clmad` cubin. Build it (needs CUDA
13.x ptxas + sm_120):

```bash
VENV=.venv/bin/python bash optim/clmad/build_ffi.sh
```

Gates/benches pick it up automatically (`field_clmad.available()`); otherwise they
use the software `field.mul`, which is byte-identical. Details in
`optim/clmad/README.md`.

## Benchmarks

The apple-to-apple comparison is **flock-zorch GPU vs unmodified flock CPU on the
same instance** (the other provers in `third_party/flock/benchmarks/` — binius64 /
plonky3 / hashcaster — are not ported here). The two halves:

- **CPU baseline** — `examples/bench_*_cpu.rs`, built with flock's own bench flags
  (thin-LTO, `codegen-units=1`, `target-cpu=native`) so it is flock's honest best on
  this box. e.g. `cargo run --release --example bench_e2e_cpu`.
- **GPU** — `python/flock_zorch/testing/*bench*.py` (driver: `bench_all.py`); e.g.
  `e2e_fused_bench.py`, `cpu_vs_gpu.py`, `e2e_{sha2,blake3,keccak3_ligerito}_bench.py`.

Upstream `flock/benchmarks` → flock-zorch mapping:

| upstream (BENCHMARKS.md) | flock-zorch apple-to-apple |
|---|---|
| Keccak-f[1600] throughput (§2) | keccak3 GPU prove vs `bench_keccak3_ligerito_cpu` |
| SHA-256 throughput (§3) | `e2e_sha2_*_bench.py` vs `bench_sha2{,_ligerito}_cpu` |
| BLAKE3 throughput (§4, §7) | `e2e_blake3_bench.py` vs `bench_blake3_cpu` |
| per-phase prover breakdown (§6) | `prover_phase_gpu_bench.py` |
| NTT / commit / sumcheck micro | `cpu_vs_gpu.py`, `commit_oracle_test.py`, `sumcheck_gpu_vs_cpu.py` |

> **Run benches on an IDLE machine.** A shared GPU (VRAM contention) or a loaded CPU
> inflates the speedup dishonestly — the CPU baseline must be flock's true best. The
> reference CPU path is **x86 scalar** (flock's NEON paths are aarch64-gated), so the
> definitive equivalence test wants flock built on Apple silicon; numbers here are
> the same-box x86 comparison. Headline results live in the top-level `README.md`.

## Updating the pins

```bash
# bump a submodule (then keep requirements.in in lockstep with the zorch pin):
git -C third_party/zorch fetch && git -C third_party/zorch checkout <commit>
git -C third_party/flock fetch && git -C third_party/flock checkout <commit>
git add third_party/zorch third_party/flock && git commit -m "deps: bump pins"
# re-verify before pushing:
scripts/dump_goldens.sh && $VENV python/flock_zorch/testing/e2e_oracle_test.py
```

## Troubleshooting

- **`CUDA_ERROR_OUT_OF_MEMORY` on a shared GPU** — set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` (above); jax otherwise grabs ~75% of VRAM up
  front and collides with other processes.
- **`ModuleNotFoundError: zorch...` or missing `zorch.hash.sha256` /
  `zorch.byte_transcript`** — the zorch submodule is on the wrong commit. It must be
  `flock-byte-fiat-shamir` @ `39396626` (`git submodule status`); `main` lacks those
  files.
- **`workspace.package.edition was not defined`** (cargo) — the `third_party` exclude
  in `Cargo.toml`'s `[workspace]` is missing; flock-core must inherit from flock's
  own workspace, not this one.
