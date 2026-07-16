# Project Context for Claude Code

Overview, setup, the reproduction path, and the benchmark all live in
[`README.md`](README.md) — start there.

## Non-negotiables

The rules every change must respect:

- **Byte-identical to flock.** Every layer is gated `frx port ≡ unmodified flock`
  over serialized bytes, anchored bottom-up (field → additive NTT → Merkle →
  zerocheck → lincheck → PCS → full `R1csProof`). The golden fixtures are dumped
  from flock-core (the `flock-core` / `flock-prover` git rev dep), so the gate
  transitively pins us to upstream. A layer is not done until its `*_oracle_test`
  is green on GPU, and no behavior change ships without its byte-match.
- **Assemble zorch's blocks, never re-implement the scheme.** The prover is built
  from zorch's scheme-agnostic spine (`Round`, Fiat-Shamir, `PCS`, fold,
  zero-check). flock-zorch adds only the flock-specific pieces the byte-match
  needs — the GHASH-basis field, the round-1 URM, the ∞-trick round loop, and
  F128↔bytes serialization — and re-derives nothing zorch already provides.
- **frx and zorch pins move in lockstep.** Bumping zorch (the `MODULE.bazel`
  `git_override`) means bumping `requirements.in`'s frx / frxlib / frx-cuda12
  wheels to the SAME version as zorch's own `requirements.in`: the binary-field
  GPU kernels must match, and CPU-only CI can't catch a desync.
