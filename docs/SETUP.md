# Running flock-zorch on a GPU box

flock-zorch is the GPU port; it depends on two pinned sibling repos:

| dep | repo | pin | how |
|---|---|---|---|
| `third_party/flock` | `succinctlabs/flock` (public) | `main` @ `73f7202` | git **submodule** — the **byte-compare oracle**: Cargo path-deps `flock-core`/`flock-prover`; `examples/dump_*.rs` dump the golden fixtures from it |
| `zorch` | `fractalyze/zorch` (private) | `main` @ `ccc7006` | bazel **`git_override`** in `MODULE.bazel` — the **scheme-agnostic spine**: `zorch.hash.sha256`, `zorch.byte_transcript` (FS duplex), the `Stage`/`Bridge`/`Round` chain roles |

> The zorch pin tracks `main` (frx-migrated). Bump it by editing the `git_override`
> commit in `MODULE.bazel` (no submodule to move); keep the `requirements.in`
> `frx`/`frxlib`/`frx-cuda12` wheels on the SAME version as zorch's own
> `requirements.in` in lockstep — the binary-field GPU kernels must be byte-identical,
> and CPU-only CI can't catch a desync. `bazel test` resolves zorch from the
> git_override; the heavy/venv gates resolve the same copy via
> `scripts/zorch_pythonpath.sh`.

## Prerequisites

- **GPU**: CUDA, sm_120 reference is an RTX 5090 (CUDA 12.9 driver stack). Any
  recent NVIDIA GPU works for the gates; the headline benches assume the 5090.
- **Rust** via rustup (`flock-core` is edition 2024).
- **Python 3.11**.
- **SSH access to `fractalyze/zorch`** (bazel's `git_override` clones it).
- Optional, for the GPU fast path: a **CUDA 13.3 `ptxas`** at `~/.local/cuda13/bin`.
  With it on `PATH` the pinned frx wheel's compiler emits the hardware `clmad`
  GF(2¹²⁸) multiply; without it everything still runs on the software
  `binary_field_ghash` multiply — byte-identical, just slower. See "clmad GPU
  acceleration" below.

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
(dumpers + CPU benches) → clmad fast-path check → regenerate the core goldens →
two **smoke gates** (`field`, `e2e`) that prove byte-identity end-to-end.

## Running the gates

The **18 core byte-identity gates** run under bazel — deps from the pip lock,
`zorch` from the `MODULE.bazel` `git_override`, goldens pulled from `//artifacts`
runfiles (dump them first, see below):

```bash
scripts/dump_goldens.sh core           # goldens the gates byte-compare against
bazel test //python:all                # all 18 (CPU; JAX_PLATFORMS=cpu + x64 pinned in .bazelrc)
bazel test //python:e2e_oracle_test    # a single gate
```

The **heavy hash-circuit gates** (keccak/sha2/blake3 — hundreds-of-MB goldens) and
the `commit` GPU perf gate are **not** bazel targets (the CUDA wheels aren't
hermetic). Run those on the venv, resolving the same git_override'd zorch via
`scripts/zorch_pythonpath.sh` (`jax_enable_x64` is set by the tests themselves):

```bash
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # don't grab ~75% of VRAM up front (shared-box friendly)
export PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)"
VENV=.venv/bin/python
scripts/dump_goldens.sh all                  # + the real hash circuits
$VENV python/flock_zorch/testing/keccak_oracle_test.py       # Keccak full prove (BaseFold)
$VENV python/flock_zorch/testing/blake3_ligerito_oracle_test.py
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
- **PCS commit** — `python/flock_zorch/testing/run_commit_gates.sh` (6 `(m,rate,batch)` configs; drives `flock_zorch/pcs/testing/commit_oracle_test.py`)
- **lincheck** — the multi-`(m,k_log,k_skip)` gate regenerates its per-config goldens.

## Optional: clmad GPU acceleration

The carryless GF(2¹²⁸) `binary_field_ghash` multiply's fast path is the hardware
PTX `clmad` (carryless multiply-add) instruction, which the pinned frx wheel's
compiler emits directly — nothing to build. Emission is gated on the runtime
`ptxas` being ≥ 13.3 (sm_120 requires it), so just put a CUDA 13.3 toolkit's
`ptxas` on `PATH`:

```bash
export PATH="$HOME/.local/cuda13/bin:$PATH"
```

With that, gates/benches use hardware `clmad` automatically; without it the wheel
keeps the software xor/shift `binary_field_ghash` path, which is byte-identical,
just slower.

## Benchmarks

The apple-to-apple comparison is **flock-zorch GPU vs unmodified flock CPU on the
same instance** (the other provers in `third_party/flock/benchmarks/` — binius64 /
plonky3 / hashcaster — are not ported here). The two halves:

- **CPU baseline** — `examples/bench_*_cpu.rs`, built with flock's own bench flags
  (thin-LTO, `codegen-units=1`, `target-cpu=native`) so it is flock's honest best on
  this box. e.g. `cargo run --release --example bench_e2e_cpu`.
- **GPU** — `python/flock_zorch/testing/*bench*.py` (driver: `bench_all.py`); e.g.
  `e2e_fused_bench.py`, `e2e_{sha2,blake3,keccak3_ligerito}_bench.py`.

Upstream `flock/benchmarks` → flock-zorch mapping:

| upstream (BENCHMARKS.md) | flock-zorch apple-to-apple |
|---|---|
| Keccak-f[1600] throughput (§2) | keccak3 GPU prove vs `bench_keccak3_ligerito_cpu` |
| SHA-256 throughput (§3) | `e2e_sha2_*_bench.py` vs `bench_sha2{,_ligerito}_cpu` |
| BLAKE3 throughput (§4, §7) | `e2e_blake3_bench.py` vs `bench_blake3_cpu` |
| per-phase prover breakdown (§6) | `prover_phase_gpu_bench.py` |
| NTT / commit / sumcheck micro | `commit_oracle_test.py`, `sumcheck_gpu_vs_cpu.py` |

> **Run benches on an IDLE machine.** A shared GPU (VRAM contention) or a loaded CPU
> inflates the speedup dishonestly — the CPU baseline must be flock's true best. The
> reference CPU path is **x86 scalar** (flock's NEON paths are aarch64-gated), so the
> definitive equivalence test wants flock built on Apple silicon; numbers here are
> the same-box x86 comparison. Headline results live in the top-level `README.md`.

## Updating the pins

```bash
# bump zorch: edit the git_override commit in MODULE.bazel (keep requirements.in in
# lockstep with the pin) — bazel re-fetches on the next command, no submodule to move:
$EDITOR MODULE.bazel                  # git_override(..., commit = "<new sha>")
# bump flock (still a submodule):
git -C third_party/flock fetch && git -C third_party/flock checkout <commit>
git add MODULE.bazel third_party/flock && git commit -m "deps: bump pins"
# re-verify before pushing:
scripts/dump_goldens.sh core && bazel test //python:all
```

## Troubleshooting

- **`CUDA_ERROR_OUT_OF_MEMORY` on a shared GPU** — set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` (above); frx otherwise grabs ~75% of VRAM up
  front and collides with other processes.
- **`ModuleNotFoundError: zorch...` or missing `zorch.hash.sha256` /
  `zorch.byte_transcript`** — the zorch submodule is on the wrong commit. It must be
  `flock-byte-fiat-shamir` @ `39396626` (`git submodule status`); `main` lacks those
  files.
- **`workspace.package.edition was not defined`** (cargo) — the `third_party` exclude
  in `Cargo.toml`'s `[workspace]` is missing; flock-core must inherit from flock's
  own workspace, not this one.
