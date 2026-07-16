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

## Native `binary_field_ghash` dtype gotchas

Compute on the dtype (`*`→clmul, `+`→XOR, `jnp.sum`→XOR-sum). The uint64[lo,hi]
lanes are the SAME 16 LE bytes, so `to_ghash`/`from_ghash` are pure bitcasts and
`ghash.tobytes()` == the wire; the proof can hold ghash and serialize directly.
- **int→ghash convert is UNIMPLEMENTED.** A ghash zero must be a bitcast of zero
  bytes (`bitcast_convert_type(jnp.zeros(2,uint64), binary_field_ghash)`), NOT
  `jnp.zeros((), binary_field_ghash)`.
- **`jnp.where` on ghash needs a concrete zeros-ARRAY default**, not a scalar —
  a scalar default emits an S64→ghash convert at compile. Mask a 0/1 select as
  `jnp.where(mask_bool, x_g, zeros_g_array)`, never `mask_uint64 * x_g`.
- **FS framing: scalar draw ≠ slice(1)** on the wire. `sample_f128()` (bare) is a
  scalar; a vector draw is `sample_f128(n)` (slice) even when n==1.
