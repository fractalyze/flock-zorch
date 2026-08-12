# Project Context for Claude Code

Overview, setup, the reproduction path, and the benchmark all live in
[`README.md`](README.md) — start there.
How the protocol is modelled — claims, roles, and what the tooling misses — is
[`docs/conventions.md`](docs/conventions.md).
How performance is measured — which numbers to trust, and the traps that have
each cost a session — is [`docs/measurement.md`](docs/measurement.md). Read it
before quoting or comparing any benchmark number.

## Non-negotiables

The rules every change must respect:

- **Byte-identical to flock.** The gate is **proof-level**: the `*_oracle_test`
  gates byte-compare every field of a full serialized proof against goldens
  dumped from flock-core (the `flock-core` / `flock-prover` git rev dep) — the
  `LigeritoProof` gate on CPU CI, the identity-e2e + hash-circuit full provers
  on GPU. One diverging byte in any layer flips every Fiat-Shamir draw after it,
  so the proof gates transitively pin FS framing, NTT, Merkle/octopus,
  zerocheck, lincheck, and ring-switch; per-layer golden gates are retired —
  don't add one, add a python-native test (no golden) for primitive behavior.
  No behavior change ships without the proof gates green (GPU set included).
  The standing GPU gates run at m=22 (ML suffix n=15): a code path gated on
  size — e.g. `_OUTER_SPLIT_MIN`'s n≥16 outer-product emission — is never
  exercised by them, so byte-gate it with an m-variant golden
  (`blake3_ligerito_golden_m32.bin` reaches n=25) before claiming green.
- **Assemble zorch's blocks, never re-implement the scheme.** The prover is built
  from zorch's scheme-agnostic spine (the `ProverRound` / `VerifierRound`
  protocols and their drivers, Fiat-Shamir, `PCS`, fold, zero-check). flock-zorch adds only the flock-specific pieces the byte-match
  needs — the GHASH-basis field, the round-1 URM, the ∞-trick round loop, and
  F128↔bytes serialization — and re-derives nothing zorch already provides.
- **frx and zorch pins move in lockstep.** Bumping zorch (the `MODULE.bazel`
  `git_override`) means bumping `requirements.in`'s frx / frxlib / frx-cuda12
  wheels to the SAME version as zorch's own `requirements.in`: the binary-field
  GPU kernels must match, and CPU-only CI can't catch a desync.
- **Prove a wheel carries the compiler fix you're waiting on before you
  validate against it.** A dev wheel is only as new as the xla commit its jax
  pin named, which can be the commit *before* the fix; the version date says
  nothing. Grep the plugin binary for a string the fix introduced —
  `strings -a .venv/lib/python3.11/site-packages/frx_plugins/xla_cuda12/xla_cuda_plugin.so
  | grep '<phrase from the fix>'` — and pair it with a control phrase that
  predates the fix, so an empty result means "absent" and not "grep is wrong".
  Skipping this reads a still-broken wheel as the fix having failed.

## Native `binary_field_ghash` dtype gotchas

Compute on the dtype (`*`→clmul, `+`→XOR, `jnp.sum`→XOR-sum). The uint64[lo,hi]
lanes are the SAME 16 LE bytes, so `to_ghash`/`from_ghash` are pure bitcasts and
`ghash.tobytes()` == the wire; the proof can hold ghash and serialize directly.
- **Ghash zeros are `jnp.zeros(n, binary_field_ghash)`** (scalar `()` and arrays
  both). The ONE exception is a `jnp.sum`/reduce over a zeros array XLA can
  *constant-fold*: the reduce identity then lowers via an unsupported S64→ghash
  convert (`UNIMPLEMENTED: Converting from S64 to BINARY_FIELD_GHASH`,
  fractalyze/jax#127). Avoid it by not feeding a reduce a fold-to-zero input — e.g.
  keccak.py skips its identically-zero r=0 RC term. Only if you truly can't, fall
  back to `bitcast_convert_type(jnp.zeros(uint64), binary_field_ghash)`.
- **0/1 select is `jnp.where(mask_bool, x_g, jnp.zeros(_, binary_field_ghash))`**,
  never `mask_uint64 * x_g` (that clmuls the mask as a field element, not a select).
- **FS framing: scalar draw ≠ slice(1)** on the wire. `sample_f128()` (bare) is a
  scalar; a vector draw is `sample_f128(n)` (slice) even when n==1.
