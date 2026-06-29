# flock-zorch — agent guide

GPU port of [succinctlabs/flock](https://github.com/succinctlabs/flock) (an
R1CS-over-GF(2) PIOP prover: zerocheck + lincheck, Ligerito/BaseFold PCS over
F₂¹²⁸) onto Fractalyze's zorch / zkx compiler stack, in the style of
`bellman-zorch` and `accumulation-zorch`. Upstream lives at `third_party/flock` — it is
the byte-compare baseline. Read `README.md` for setup/run.

## Non-negotiables

### 1. Byte-identical oracle gates are mandatory
Every layer ships with a byte-match against **unmodified** flock. The jax port's
serialized output must equal flock-core's reference bytes, anchored bottom-up:
**field → additive-NTT → Merkle → zerocheck → lincheck → PCS → e2e proof.**
A layer is not "done" until its `*_oracle_test` is green on GPU. Fixtures are
dumped from flock-core itself (`examples/dump_*.rs`), so the gate transitively
pins us to upstream. (cf. accumulation-zorch's three-way oracle.)

### 2. Arithmetic in flock's GHASH basis, directly — NOT zk_dtypes binary fields
`zk_dtypes.binary_field_t7` is GF(2¹²⁸) in the `x²+x+α` **tower** basis; flock
uses the **GHASH** basis (p(x)=x¹²⁸+x⁷+x²+x+1, const 0x87). Isomorphic but NOT
bit-compatible (`2·2 = 3` tower vs `4` GHASH — verified empirically). flock hashes
raw field bytes everywhere (Merkle leaves, SHA-256 transcript), so byte-identity
demands flock's basis. We implement GF(2¹²⁸) over `uint64` lanes (`field.py`) —
bytes match flock by construction. Verified on GPU: `x·x¹²⁷ = 0x87`.

### 3. Sequential transcript on host; bulk arithmetic on device
flock's Fiat-Shamir `Challenger` is a **SHA-256** duplex hash chain (NOT BLAKE3 —
the upstream section comment is stale) — strictly sequential, runs on the host.
Between challenges the bulk work (NTT butterflies, sumcheck folds, Merkle build,
R1CS matvec over 2^m elements) is data-parallel and fused into one `.mlirbc` per
round-group on the GPU. Per-round host↔device traffic is tiny (a couple F128 + a
32-byte root); the loop is latency-bound on round count, so keep bulk state
resident on device. Proof bytes = bincode 1.3 **fixint-LE** + 7-byte `FLOCK`+ver+flavor
header (the "varint" upstream comment is stale).

## Performance is the zkx compiler's job
Keep `python/flock_zorch` readable. The `ghash_mul` carryless product is a plain
64-step bit reduction — correct but slow (there is **no PTX CLMUL** instruction;
confirmed absent from the PTX ISA + the in-tree LLVM). Optimization happens in the
zkx / prime-ir compiler, gated by the same byte-match:
- prime-ir already has carryless-mul lowerings but **CPU-only** (SpecializeBinaryFieldToX86 = `vpclmulqdq`, ToARM = `pmull`); the **GPU pipeline never runs a binary-field pass**.
- Extension point: `zkx/backends/gpu/codegen/emitters/emitter_base.cc::AddLoweringPasses` — wire prime-ir's portable `BinaryFieldToArith` (shift-XOR/Karatsuba) into the GPU path, and/or a fused `flock.ghash_mul` composite emitter (the zorch `fused_region` name-routed mechanism).

## Dependency on `zorch`
Scheme-agnostic machinery lives upstream in `zorch` (`third_party/zorch`), per the team
rule (sp1-zorch/whir-zorch do the same). flock-zorch reuses:
- `zorch.byte_transcript.Sha256Transcript` — the Merlin-over-SHA256 Fiat-Shamir
  duplex; `challenger.py` is the thin F128 (16-byte lo‖hi) glue over it.
- `zorch.hash.sha256` — the byte-SHA-256 (data-parallel device sibling of the
  transcript's host hashlib path).
- (planned) `zorch.sumcheck.field_ops.FieldOps` — the binary-field sumcheck seam.
Run gates with **`PYTHONPATH=python:third_party/zorch`** so the `zorch` package resolves.
What stays flock-specific: F128↔bytes serialization, the round-1 URM (F8/φ8/F8-NTT),
the ∞-trick round loop, and the `prove_packed` assembly.

## Layout
- `python/flock_zorch/` — the jax prover port (authoring surface).
- `python/flock_zorch/testing/` — per-layer KAT + byte-match gates.
- `export/export_*.py` — fuse a core → one StableHLO `.mlirbc` (PJRT path; WIP).
- `examples/dump_*.rs` — golden-byte dumpers from flock-core (the oracle source).
- `src/`, `crates/zkx-pjrt/` — thin Rust host driver + PJRT FFI shim (WIP; copy
  `crates/zkx-pjrt` verbatim from `../bellman-zorch` — it already carries
  `BINARY_FIELD_T0..T7` buffer tags).

## Toolchain
- GPU: RTX 5090 (sm_120, Blackwell), CUDA 12.9. zorch venv:
  `scripts/setup.sh` builds `./.venv` from `requirements.in` (the jax_fork
  jax-cuda12 stack `0.10.0.dev*` + `jax-cuda12-pjrt` + `zk_dtypes` 0.0.7 — mirror
  zorch's pin). `jax_enable_x64` required for the uint64 lanes.
- Rust: standalone rustup in `~/.cargo` (flock is edition 2024). `flock-core` is a
  path dep at `third_party/flock/crates/flock-core`.
