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
  **Exception — the BLAKE3 Fiat-Shamir arm has no proof golden and cannot get
  one cheaply.** The `flock-core` / `flock-prover` deps are UPSTREAM
  `succinctlabs/flock`, which has no BLAKE3 challenger, so every in-tree golden
  is SHA-256-FS (`blake3_ligerito_golden.bin` is the blake3 *circuit* proved
  with the SHA-256 challenger — the name misleads). That arm's absolute
  reference is the hand-pasted `Layr-Labs/flock-challenge` fixtures in
  `testing/blake3_challenger_test.py`; its prove path rides the SHA-256
  goldens, which pin the call sites because a `ProveProfile` only swaps which
  transcript they talk to. A real blake3-FS golden needs the fork added as a
  second Cargo dep plus a dumper written against it — plan for that, don't
  assume `dump_goldens.sh` can be pointed at it.
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
  **When the fix introduces no new string, this test cannot work** — a
  pure-performance change (prime-ir#432 rewrote GHASH clmul as integer
  multiplies) leaves the symbol names byte-identical, so `clmul` and
  `select_xor` grep the same with and without it. Then the only reliable
  check is behavioural: A/B the same benchmark against a known-good plugin
  kept beside the new one. Keep the old dylib rather than overwriting it.
  **But never A/B two selfbuilt plugins built from different base commits.** A
  selfbuilt plugin encodes its whole source tree, not just the change you are
  testing, and a day of unrelated commits is easily worth more than the change.
  This has already produced one confidently-reported +46% that evaporated when
  the two arms were rebuilt from the same base (fractalyze/xla#528). Either
  rebuild both arms from one base, or — much cheaper — drive the change from a
  runtime knob and leave the binary fixed: `xla_gpu_max_fusion_ir_size`
  (fractalyze/xla#510) is the model. When `XLA_FLAGS` rejects a flag your
  selfbuilt plugin knows but the installed frx wheel does not, pass it as
  `frx.jit(..., compiler_options={...})`, which reaches the plugin's compiler
  directly — JAX allows it on the top-level jit only, and nested jits inline
  into the parent's module so the parent's option already covers them.
  The Metal plugin has no published wheel at all — the lockstep set is
  `frx/frxlib/frx-cuda12-*` — so `frx_plugins/xla_metal/*.dylib` is
  structurally selfbuilt, and rebuilding it without the same
  `--override_repository=` flags silently drops whatever unlanded fix it
  carried. A regression whose size matches a known lever's is that, not your
  change. **On macOS/arm64 `frxlib` goes further: only the releases are
  published, not the dev snapshots this file pins**, so
  `pip install -r requirements.in` cannot resolve and the local Metal rig
  cannot follow main from wheels at all. Build `frx` + `frxlib` +
  `frx-metal-pjrt` together from the one jax commit whose build timestamp
  matches the pinned version (`build/build.py build --wheels=...` with
  `--override_repository=xla=` at that commit's `XLA_COMMIT`, then
  `build/frx_rename.py`), and install every wheel with `--no-deps` — a
  plain `pip install hash-frx` re-resolves `frx` off the index and undoes
  the set. Install the plugin as the `frx-metal-pjrt` **wheel**, never by
  copying the dylib: `metal_plugin_extension.so` and `version.py` must move
  with it or they mismatch the next time frxlib's version changes.
- **A `fused_region` decomposition is a fallback, not the code that runs.**
  `ZorchFusedRegionRewriter` routes the composite to its kCustom emitter on
  **Metal as well as CUDA** (`MetalCompiler : GpuCompiler` does not override the
  pipeline), so optimizing the Python body — `_urm._round1_partial_decomp` and
  friends — moves nothing on device. Before profiling a fused region, confirm
  what it lowered to: `frx.jit(fn).lower(...).compile().as_text()` gives the
  optimized HLO (`XLA_FLAGS` dumps only the input module at PJRT level), and the
  fix then belongs in `xla/backends/gpu/codegen/emitters/`.

## Porting a witness layout from flock

The four hash circuits live together in `python/flock_zorch/r1cs_hashes/`,
mirroring flock's own `flock-prover/src/r1cs_hashes/`; that package's
`__init__.py` is the map of which module carries what.

- **The `*_BASE` / `*_POS` constants are the spec; the `//!` header diagram is
  not.** They disagree in `r1cs_hashes/sha2.rs` at the pinned rev (the header
  puts `Z_CONST` at bit 512, the constants at 31,400 — so every field from `M`
  on is off by one if you trust the header). `USEFUL_BITS` can match while every
  interior boundary is wrong, so agreeing totals prove nothing.
- **Check a transcription against the golden before writing device code.** The
  goldens carry z/a/b, so a numpy reference built from the field list can be
  compared directly in seconds. A mismatch then names a region instead of
  surfacing as "the proof diverges" after a kernel exists.
- **The packing model differs by circuit family.** blake3 and sha2 pack fields
  tightly, so values straddle 64-bit words and every field needs shift
  arithmetic — both go through `r1cs_hashes.common.emit`, which addresses
  fields by bit offset rather than walking a cursor, because neither circuit
  produces its fields in bit order. keccak and keccak3 place each state in a
  2,048-bit aligned slot and write whole u64 lanes, so their "field list" is a
  lane map with no shift arithmetic at all. Parameterizing by `K_LOG` alone does
  not cover this.

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
