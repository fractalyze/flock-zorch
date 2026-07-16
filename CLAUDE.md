# flock-zorch — agent guide

GPU port of [succinctlabs/flock](https://github.com/succinctlabs/flock) (an
R1CS-over-GF(2) PIOP prover: zerocheck + lincheck, Ligerito/BaseFold PCS over
F₂¹²⁸) onto Fractalyze's zorch / frx compiler stack, in the style of
`bellman-zorch` and `accumulation-zorch`. Upstream lives at `third_party/flock` —
it is the byte-compare baseline.

This file is agent conventions only; everything else lives in the docs:
- [`README.md`](README.md) — what/why, architecture, benchmarks, byte-identity
  status, setup, toolchain.
- [`docs/SETUP.md`](docs/SETUP.md) — prerequisites, gate environment, pins and
  bumping them, golden regeneration, `clmad` GPU acceleration.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — methodology and measurements.

**Comments: terse.** Prefer none over verbose — a good step/target name or
self-evident code beats a paragraph. Drop a comment rather than pad it.

## Non-negotiable: byte-identical oracle gates are mandatory

Every layer ships with a byte-match against **unmodified** flock. The frx port's
serialized output must equal flock-core's reference bytes, anchored bottom-up:
**field → additive-NTT → Merkle → zerocheck → lincheck → PCS → e2e proof.**
A layer is not "done" until its `*_oracle_test` is green on GPU. Fixtures are
dumped from flock-core itself (`examples/dump_*.rs`), so the gate transitively
pins us to upstream. Core gates: `bazel test //python:all`; the heavy/GPU gates
run on the venv (`docs/SETUP.md`).

Fixtures must end in `_golden.bin` — `artifacts/BUILD.bazel` globs `*_golden.bin`
into every gate's runfiles, so qualify a config variant *before* the suffix
(`basefold_3epoch_golden.bin`, not `basefold_golden_3epoch.bin`, which the glob
silently drops). A non-default config (e.g. a multi-epoch anchor) is dumped by an
explicit `dump_<x> <args> artifacts/<name>_golden.bin` line in
`scripts/dump_goldens.sh`, mirroring the `run_commit_gates.sh` / lincheck sweeps.
