# Project Context for Claude Code

Overview, setup, the reproduction path, and the benchmark all live in
[`README.md`](README.md) — start there.

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
- **Assemble zorch's blocks, never re-implement the scheme.** The prover is built
  from zorch's scheme-agnostic spine (`Round`, Fiat-Shamir, `PCS`, fold,
  zero-check). flock-zorch adds only the flock-specific pieces the byte-match
  needs — the GHASH-basis field, the round-1 URM, the ∞-trick round loop, and
  F128↔bytes serialization — and re-derives nothing zorch already provides.
- **frx and zorch pins move in lockstep.** Bumping zorch (the `MODULE.bazel`
  `git_override`) means bumping `requirements.in`'s frx / frxlib / frx-cuda12
  wheels to the SAME version as zorch's own `requirements.in`: the binary-field
  GPU kernels must match, and CPU-only CI can't catch a desync.
  A `.bazelrc.user` `--override_module=zorch=<local checkout>` makes that
  checkout — not the MODULE pin — the effective zorch, so after a pin bump also
  advance the checkout and re-`pip install -r requirements.in` the venv.
  Skipping either leaves two frx versions in the test runfiles with sys.path
  order deciding which wins per test; the symptom is a `TypeError: ... got an
  unexpected keyword argument` on an API only the newer wheel has, on tests
  that are green in CI.
- **flock code is hardware-agnostic.** No `frx.default_backend()` or
  dtype-sniffing backend branches in `python/flock_zorch/` outside `testing/`
  — per-backend dispatch lives inside zorch's kernels
  (`zorch.utils.binary_field`, `zorch.pcs.ring_switch`), which each carry a
  Pallas GPU path and a portable XLA CPU path. If a zorch kernel lacks a
  portable path, add it in zorch; don't fork a local fallback here (#165).

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
