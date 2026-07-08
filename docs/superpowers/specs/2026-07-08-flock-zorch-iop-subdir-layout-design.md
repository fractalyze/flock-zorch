# flock_zorch — IOP-grouped subdirectory layout

**Status:** design, pending review
**Date:** 2026-07-08

## Motivation

`python/flock_zorch/` is a flat package: 24 source modules plus one central
`testing/` directory holding ~35 oracle tests and benches. As the port has
grown (field → additive-NTT → Merkle → zerocheck → lincheck → PCS → e2e), the
flat layout no longer signals which files belong together. The sibling repos
`zorch` and `sp1_zorch` group related IOPs into subdirectories, each with its
own `testing/`. This spec brings `flock_zorch` to the same shape.

Goal: group the 24 modules into six IOP subdirectories that mirror the protocol
layering, colocate each group's tests, and rename files to remove the
redundancy that namespacing exposes — **without changing any behavior**. The
mandatory byte-identical oracle gates (CLAUDE.md non-negotiable) are the
correctness contract: the reorg is done only when the gates are green.

## Reference conventions (from `zorch`)

- Cross-cutting primitives live at the **top level**: `zorch/transcript.py`,
  `prove.py`, `round.py`, `verify.py`. flock_zorch's analogues — `challenger.py`
  (Fiat-Shamir, = `transcript.py`) and the e2e `prover.py` (= `prove.py`) —
  therefore stay at the top level.
- Each IOP group is a package with `__init__.py`, its modules, `testing/`, and
  (in zorch) a `BUILD.bazel`.
- Group modules use **descriptive content names**, not forced roles:
  `coding/reed_solomon.py`, `coding/linear_code.py`. `prover.py`/`verifier.py`
  appear only where a protocol has those roles (`sumcheck/prover.py`).

## Target structure

```
flock_zorch/
├── __init__.py
├── prover.py            # e2e driver           (top-level, = zorch prove.py)   [unchanged]
├── challenger.py        # Fiat-Shamir SHA256   (top-level, = zorch transcript.py) [unchanged]
├── testing/             # shared _util.py + __init__.py + all full-stack / e2e tests
├── field/
│   ├── __init__.py      # re-exports the public F128 API
│   ├── f128.py          ← field.py
│   ├── gf8.py
│   ├── _gf8_device.py
│   ├── _hostfield.py
│   └── testing/
├── hash/
│   ├── __init__.py
│   ├── merkle.py
│   ├── sha256.py
│   └── testing/
├── sumcheck/
│   ├── __init__.py      # re-exports build_eq, fold_*, eq_eval, ONE
│   ├── eq.py            ← sumcheck.py
│   └── testing/
├── zerocheck/
│   ├── __init__.py      # re-exports prove, _lagrange_weights, ...
│   ├── prover.py        ← zerocheck.py
│   ├── _fold.py         ← _zerocheck_fold.py
│   └── testing/
├── lincheck/
│   ├── __init__.py
│   ├── prover.py        ← lincheck.py
│   ├── chain.py
│   ├── keccak.py        ← keccak_lincheck.py
│   ├── keccak3.py       ← keccak3_lincheck.py
│   ├── _csc_fold.py
│   └── testing/
└── pcs/
    ├── __init__.py      # re-exports FlockPcsProver, FlockPcsProverData
    ├── prover.py        ← pcs.py
    ├── commit.py        ← pcs_commit.py
    ├── open.py          ← pcs_open.py
    ├── basefold.py
    ├── fri.py
    ├── ring_switch.py
    ├── ligerito.py      ← zorch_ligerito.py
    └── testing/
```

### Naming rule (settled)

- **Math-primitive groups** use descriptive content names: `field/f128.py`
  (the field is F₂¹²⁸), `sumcheck/eq.py` (eq/fold multilinear utilities — this
  module builds `eq` tables and folds; it is not a sumcheck *prover*).
- **IOP-protocol groups** name their main module `prover.py`
  (`zerocheck/prover.py`, `lincheck/prover.py`, `pcs/prover.py`). flock_zorch is
  a prover-only port (byte-compared against flock's prover), so there is no
  `verifier.py` sibling — the absence is consistent across all three.
- **Helpers / leaf modules** keep descriptive names (`_fold.py`, `chain.py`,
  `keccak.py`, `commit.py`, `open.py`, `basefold.py`, `fri.py`,
  `ring_switch.py`, `ligerito.py`).
- Each group `__init__.py` **re-exports its public API** (including the
  cross-module `_`-prefixed helpers other groups import, e.g.
  `zerocheck._lagrange_weights`, `field._to_int`) so existing
  `from flock_zorch import <group>` and `from flock_zorch.<group> import <name>`
  call sites resolve unchanged.
- `pcs/open.py` keeps the name `open`; the module defines a function `open()`.
  To avoid shadowing the `open` builtin, importers alias:
  `from flock_zorch.pcs import open as pcs_open` (preserves the existing
  `pcs_open.open(...)` call form byte-for-byte).

### Source move + rename table

| current                    | new                          |
|----------------------------|------------------------------|
| `field.py`                 | `field/f128.py`              |
| `gf8.py`                   | `field/gf8.py`               |
| `_gf8_device.py`           | `field/_gf8_device.py`       |
| `_hostfield.py`            | `field/_hostfield.py`        |
| `merkle.py`                | `hash/merkle.py`             |
| `sha256.py`                | `hash/sha256.py`             |
| `sumcheck.py`              | `sumcheck/eq.py`             |
| `zerocheck.py`             | `zerocheck/prover.py`        |
| `_zerocheck_fold.py`       | `zerocheck/_fold.py`         |
| `lincheck.py`              | `lincheck/prover.py`         |
| `chain.py`                 | `lincheck/chain.py`          |
| `keccak_lincheck.py`       | `lincheck/keccak.py`         |
| `keccak3_lincheck.py`      | `lincheck/keccak3.py`        |
| `_csc_fold.py`             | `lincheck/_csc_fold.py`      |
| `pcs.py`                   | `pcs/prover.py`              |
| `pcs_commit.py`            | `pcs/commit.py`              |
| `pcs_open.py`              | `pcs/open.py`                |
| `basefold.py`              | `pcs/basefold.py`            |
| `fri.py`                   | `pcs/fri.py`                 |
| `ring_switch.py`           | `pcs/ring_switch.py`         |
| `zorch_ligerito.py`        | `pcs/ligerito.py`            |
| `prover.py`                | `prover.py` (unchanged)      |
| `challenger.py`            | `challenger.py` (unchanged)  |

### Test placement

A test's home is the group of the module it exercises. Full-stack tests (those
that import `prover` and run an e2e proof) stay at the top level — they gate the
whole pipeline, not one group. This mirrors zorch, whose top-level `testing/`
holds the `prove`/`transcript`/`round` tests. Note the hash-circuit sources
(`keccak.py`, `keccak3.py`) live in `lincheck/`; their **e2e** tests
(`keccak_oracle_test`, etc.) run the full prover and so live at the top level.

| group `testing/`      | tests                                                                                                                                                                       |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| top-level `testing/`  | `e2e_oracle_test`, `challenger_oracle_test`, `keccak_oracle_test`, `keccak_ligerito_oracle_test`, `keccak_chain_oracle_test`, `keccak3_ligerito_oracle_test`, `sha2_oracle_test`, `sha2_ligerito_oracle_test`, `blake3_oracle_test`, `blake3_ligerito_oracle_test`, `e2e_fused_bench`, `e2e_gpu_bench`, `e2e_blake3_bench`, `e2e_sha2_bench`, `e2e_sha2_ligerito_bench`, `e2e_keccak3_ligerito_bench`, `bench_all`, `prover_phase_gpu_bench`, plus shared `__init__.py` + `_util.py` |
| `field/testing/`      | `gf8_urm_oracle_test`                                                                                                                                                      |
| `hash/testing/`       | `merkle_oracle_test`, `merkle_multi_oracle_test`, `merkle_openings_test`, `sha256_oracle_test`                                                                             |
| `sumcheck/testing/`   | `sumcheck_oracle_test`, `sumcheck_gpu_vs_cpu`                                                                                                                              |
| `zerocheck/testing/`  | `zerocheck_oracle_test`                                                                                                                                                    |
| `lincheck/testing/`   | `lincheck_oracle_test`, `lincheck_circuit_protocol_test`, `chain_shift_oracle_test`                                                                                        |
| `pcs/testing/`        | `commit_oracle_test`, `pcs_open_oracle_test`, `pcs_seam_oracle_test`, `basefold_oracle_test`, `basefold_verify_oracle_test`, `ring_switch_oracle_test`, `row_batch_oracle_test`, `ligerito_oracle_test`, `zorch_ligerito_driver_oracle_test`, `zorch_ligerito_fs_oracle_test`, `coding_oracle_test` |

Each new `testing/` dir gets an `__init__.py`. The shared `_util.py` stays at
`flock_zorch/testing/_util.py`; group tests import it via
`from flock_zorch.testing._util import ...` (unchanged path).

## Import rewrite

~122 intra-repo import sites (38 in source, 84 in tests) change from the flat
path to the grouped path. Mechanical mapping:

| old import fragment            | new import fragment                    |
|--------------------------------|----------------------------------------|
| `flock_zorch.gf8`              | `flock_zorch.field.gf8`                |
| `flock_zorch._gf8_device`      | `flock_zorch.field._gf8_device`        |
| `flock_zorch._hostfield`       | `flock_zorch.field._hostfield`         |
| `flock_zorch.merkle`           | `flock_zorch.hash.merkle`              |
| `flock_zorch.sha256`           | `flock_zorch.hash.sha256`              |
| `flock_zorch.sumcheck`         | `flock_zorch.sumcheck` (pkg; `eq.py` behind re-export) |
| `flock_zorch._zerocheck_fold`  | `flock_zorch.zerocheck._fold`          |
| `flock_zorch._csc_fold`        | `flock_zorch.lincheck._csc_fold`       |
| `flock_zorch.chain`            | `flock_zorch.lincheck.chain`           |
| `flock_zorch.keccak_lincheck`  | `flock_zorch.lincheck.keccak`          |
| `flock_zorch.keccak3_lincheck` | `flock_zorch.lincheck.keccak3`         |
| `flock_zorch.pcs` (module)     | `flock_zorch.pcs` (pkg; `prover.py` behind re-export) |
| `flock_zorch.pcs_commit`       | `flock_zorch.pcs.commit`               |
| `flock_zorch.pcs_open`         | `flock_zorch.pcs.open` (alias `as pcs_open`) |
| `flock_zorch.basefold`         | `flock_zorch.pcs.basefold`             |
| `flock_zorch.fri`              | `flock_zorch.pcs.fri`                  |
| `flock_zorch.ring_switch`      | `flock_zorch.pcs.ring_switch`          |
| `flock_zorch.zorch_ligerito`   | `flock_zorch.pcs.ligerito`             |
| `flock_zorch.field`            | `flock_zorch.field` (pkg; `f128.py` behind re-export) |
| `flock_zorch.challenger`       | `flock_zorch.challenger` (unchanged)   |
| `flock_zorch.zerocheck`        | `flock_zorch.zerocheck` (pkg; `prover.py` behind re-export) |
| `flock_zorch.lincheck`         | `flock_zorch.lincheck` (pkg; `prover.py` behind re-export) |

`from flock_zorch import X` grouped-package forms keep working because the group
`__init__.py` re-exports. Bare-name forms of moved leaf modules
(`from flock_zorch import merkle`) become `from flock_zorch.hash import merkle`.

## BUILD.bazel

Keep a single `python/BUILD.bazel`; switch its globs from flat to recursive.

- `flock_zorch` `py_library`: `srcs = glob(["flock_zorch/**/*.py"], exclude =
  ["flock_zorch/**/testing/**"])`. `imports = ["."]` unchanged, so
  `flock_zorch.pcs.prover` resolves. External `deps` unchanged.
- `test_support` `py_library`: unchanged (`flock_zorch/testing/__init__.py` +
  `_util.py`).
- Globbed `py_test` set: `glob(["flock_zorch/**/testing/*_oracle_test.py"],
  exclude = ["flock_zorch/**/testing/%s_oracle_test.py" % h for h in _HEAVY])`.
  Test `name` is now the **basename** (`_f.rpartition("/")[2][:-len(".py")]`)
  rather than a fixed-prefix strip; `_HEAVY` matches on basename, unchanged.
- The two non-`*_oracle_test` pure-host `py_test`s (`merkle_openings_test`,
  `lincheck_circuit_protocol_test`) keep explicit rules, pointed at their new
  paths (`flock_zorch/hash/testing/...`, `flock_zorch/lincheck/testing/...`).

`//python:all` and every `bazel test //python:<name>` target label is preserved
(names are still basenames), so the gate scripts need no target renames.

**Out of scope (possible follow-up):** splitting into per-subdir `BUILD.bazel`
files with per-group `py_library` targets and inter-group `deps` (the fully
zorch-faithful form). It is higher-risk wiring against the byte-gate and buys
nothing for this reorg; deferred.

## External touch-ups

- `python/flock_zorch/testing/run_commit_gates.sh` — repoint any module/test
  paths it references (commit gate now `flock_zorch/pcs/testing/commit_oracle_test`).
- `docs/SETUP.md` — the `python/flock_zorch/testing/run_commit_gates.sh` path is
  unchanged, but the "PCS commit" test reference should note the new
  `pcs/testing/` home.
- `README.md` / `CLAUDE.md` — no path-specific references found; re-scan during
  implementation and fix any that surface.

## Verification strategy

Behavior must not change; the oracle gates are the proof.

1. **Staged, group by group.** Move one group (source + its imports + its
   `testing/` + BUILD glob), then run that group's gates green before the next.
   Order bottom-up along the dependency DAG: `field` → `hash` → `sumcheck` →
   `zerocheck` → `lincheck` → `pcs` → top-level `testing/`.
2. **Per stage:** `bazel test //python:all` for the CPU gates; the heavy/GPU
   gates (`_HEAVY`) per `docs/SETUP.md` on the venv.
3. **Final:** full `bazel test //python:all` + the heavy/GPU suite green; a
   byte-diff of a sample proof before/after confirms identity.

## Risks

- **Import cycles surfaced by re-export `__init__.py`.** The current flat
  modules already import lazily where cycles exist (`pcs_open` does a local
  `import ring_switch, basefold`). Preserve those local imports verbatim.
- **`open` builtin shadow** in `pcs/` — mitigated by the `as pcs_open` alias
  rule above; verify no file in `pcs/` calls the builtin `open()`.
- **Missed import site.** A single stale `flock_zorch.<old>` import fails at
  import time; the gates catch it. Grep for the flat names after each stage.
- **Golden fixtures / `data` deps** (`//artifacts:goldens`) are unaffected —
  test targets keep `data = ["//artifacts:goldens"]`.

## Out of scope

- Per-subdir `BUILD.bazel` files (deferred, see BUILD section).
- Any logic, algorithm, or serialization change. This is a pure relocate +
  rename + import-rewrite.
