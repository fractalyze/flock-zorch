# flock-zorch — agent guide

GPU port of [succinctlabs/flock](https://github.com/succinctlabs/flock) (an
R1CS-over-GF(2) PIOP prover: zerocheck + lincheck, Ligerito/BaseFold PCS over
F₂¹²⁸) onto Fractalyze's zorch / frx compiler stack, in the style of
`bellman-zorch` and `accumulation-zorch`. Upstream flock is the byte-compare
baseline — pinned as the `flock-core` / `flock-prover` git rev dep in `Cargo.toml`.

This file is agent conventions only; everything else lives in
[`README.md`](README.md) — what/why, architecture, setup, gates, benchmarks,
and toolchain.

**Comments: terse.** Prefer none over verbose — a good step/target name or
self-evident code beats a paragraph. Drop a comment rather than pad it.

## Non-negotiable: byte-identical oracle gates are mandatory

Every layer ships with a byte-match against **unmodified** flock. The frx port's
serialized output must equal flock-core's reference bytes, anchored bottom-up:
**field → additive-NTT → Merkle → zerocheck → lincheck → PCS → e2e proof.**
A layer is not "done" until its `*_oracle_test` is green on GPU. Fixtures are
dumped from flock-core itself (`examples/dump_*.rs`), so the gate transitively
pins us to upstream. Core gates: `bazel test //python:all`; the heavy/GPU gates
run on the venv (see README "Reproduce").
