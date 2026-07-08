# flock_zorch IOP-Grouped Subdirectory Layout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the flat `python/flock_zorch/` package into six IOP subpackages (`field/`, `hash/`, `sumcheck/`, `zerocheck/`, `lincheck/`, `pcs/`) with colocated `testing/` dirs, mirroring `zorch`/`sp1_zorch`, with **no behavior change**.

**Architecture:** Pure relocate + rename + import-rewrite. `git mv` preserves history. Cross-cutting primitives (`prover.py` e2e driver, `challenger.py` Fiat-Shamir) stay top-level (= zorch's `prove.py`/`transcript.py`). Each subpackage `__init__.py` re-exports its public API so `from flock_zorch import <group>` call sites survive. A single `python/BUILD.bazel` keeps working via recursive globs. Correctness is proven by the existing byte-identical oracle gates staying green after each group moves.

**Tech Stack:** Python (jax/numpy), Bazel (`rules_python`), consumed off `PYTHONPATH`.

**Spec:** `docs/superpowers/specs/2026-07-08-flock-zorch-iop-subdir-layout-design.md`

## Global Constraints

- **No logic/serialization change.** This is a move+rename+import-rewrite only. Any diff to a function body (beyond an import line) is a bug.
- **Byte-identical oracle gates are the test** (CLAUDE.md non-negotiable). A task is done only when its gates are green: `bazel test //python:all` for CPU gates; heavy/GPU gates (the `_HEAVY` list) run on the venv per `docs/SETUP.md`.
- **Preserve import statement positions verbatim** — several modules use late/local imports to break pre-existing cycles (e.g. `gf8.py:199 import _gf8_device`, `pcs_open.open()`'s local `import ring_switch, basefold`). Only change the dotted path, never the line's position or laziness.
- **Use `git mv`** for every file move (history + rename detection).
- **Gates run with** `PYTHONPATH=python:../zorch` (per `challenger.py` docstring) when invoked outside Bazel.
- No `Co-Authored-By` trailer on commits (repo/user convention). Commit prefix: `refactor(<group>):`.

---

## Master import-rewrite mapping

Every task applies this table. **Rule:** package-preserved groups keep `from flock_zorch import <group>`; moved **leaf** modules get `from flock_zorch.<group> import <leaf>`, aliased back to the old local name (`as <old>`) **only** where the file was renamed, so call sites stay byte-identical. Split multi-name `from flock_zorch import a, b, c` lines: keep package-preserved names on the original line, add a new line per moved leaf.

| old local name (`from flock_zorch import …` / `flock_zorch.<x>`) | new import line |
|---|---|
| `field` (attr access `field.X`) | `from flock_zorch import field` *(now a subpackage; re-exported)* |
| `sumcheck` | `from flock_zorch import sumcheck` *(subpackage; re-exported)* |
| `zerocheck` | `from flock_zorch import zerocheck` *(subpackage; re-exported)* |
| `lincheck` | `from flock_zorch import lincheck` *(subpackage; re-exported)* |
| `challenger`, `prover` | unchanged (top-level modules) |
| `gf8` | `from flock_zorch.field import gf8` |
| `_gf8_device` | `from flock_zorch.field import _gf8_device` |
| `_hostfield` | `from flock_zorch.field import _hostfield` *(keep any `as hf`)* |
| `merkle` | `from flock_zorch.hash import merkle` |
| `sha256` | `from flock_zorch.hash import sha256` |
| `chain` | `from flock_zorch.lincheck import chain` |
| `keccak_lincheck` | `from flock_zorch.lincheck import keccak as keccak_lincheck` |
| `keccak3_lincheck` | `from flock_zorch.lincheck import keccak3 as keccak3_lincheck` |
| `pcs` (module `FlockPcsProver`) | `from flock_zorch.pcs import FlockPcsProver, FlockPcsProverData` *(or `from flock_zorch import pcs`)* |
| `pcs_commit` | `from flock_zorch.pcs import commit as pcs_commit` |
| `pcs_open` | `from flock_zorch.pcs import open as pcs_open` |
| `basefold` | `from flock_zorch.pcs import basefold` |
| `fri` | `from flock_zorch.pcs import fri` |
| `ring_switch` | `from flock_zorch.pcs import ring_switch` |
| `zorch_ligerito` | `from flock_zorch.pcs import ligerito as zorch_ligerito` |

**Intra-group sibling imports** use the fully-qualified submodule path (not the package), to stay robust against `__init__` init-order: e.g. inside `zerocheck/prover.py`, `from flock_zorch.zerocheck._fold import …`; inside `lincheck/chain.py`, `from flock_zorch.lincheck.prover import _round_eval, _bind_top`.

---

### Task 1: Switch `python/BUILD.bazel` to recursive globs (no file moves yet)

Prepares the build so moved files are still collected. With no files moved, `**` matches the same flat files → gates stay green. This isolates the build change from the moves.

**Files:**
- Modify: `python/BUILD.bazel`

- [ ] **Step 1: Baseline — confirm gates are green before any change**

Run: `cd /home/ryan/Workspace/flock-zorch2 && bazel test //python:all`
Expected: all PASS (record the count).

- [ ] **Step 2: Edit the `flock_zorch` library srcs glob to recurse**

In `python/BUILD.bazel`, change:
```python
    srcs = glob(["flock_zorch/*.py"]),
```
to:
```python
    srcs = glob(
        ["flock_zorch/**/*.py"],
        exclude = ["flock_zorch/**/testing/**"],
    ),
```

- [ ] **Step 3: Edit the globbed `py_test` block to recurse + derive names from basename**

Change the comprehension's glob + name expression:
```python
[
    py_test(
        name = _f.rpartition("/")[2][:-len(".py")],
        size = "medium",  # slowest gate ~64s (ligerito)
        srcs = [_f],
        data = ["//artifacts:goldens"],
        deps = [
            ":flock_zorch",
            ":test_support",
        ],
    )
    for _f in glob(
        ["flock_zorch/**/testing/*_oracle_test.py"],
        exclude = ["flock_zorch/**/testing/%s_oracle_test.py" % _h for _h in _HEAVY],
    )
]
```
(`_HEAVY` is unchanged — it now matches on basename via the `**/testing/` prefix.)

- [ ] **Step 4: Run gates to confirm the glob change is a no-op**

Run: `bazel test //python:all`
Expected: identical PASS set to Step 1.

- [ ] **Step 5: Commit**

```bash
git add python/BUILD.bazel
git commit -m "refactor(build): recurse flock_zorch source/test globs for subpackage layout"
```

---

### Task 2: `field/` subpackage

**Files:**
- Create: `flock_zorch/field/__init__.py`, `flock_zorch/field/testing/__init__.py`
- Move: `field.py→field/f128.py`, `gf8.py→field/gf8.py`, `_gf8_device.py→field/_gf8_device.py`, `_hostfield.py→field/_hostfield.py`, `testing/gf8_urm_oracle_test.py→field/testing/`

- [ ] **Step 1: Create the package dirs + move files**

```bash
cd /home/ryan/Workspace/flock-zorch2/python/flock_zorch
mkdir -p field/testing
git mv field.py field/f128.py
git mv gf8.py field/gf8.py
git mv _gf8_device.py field/_gf8_device.py
git mv _hostfield.py field/_hostfield.py
git mv testing/gf8_urm_oracle_test.py field/testing/gf8_urm_oracle_test.py
```

- [ ] **Step 2: Write `field/__init__.py` (re-export public + cross-module private API)**

```python
"""F₂¹²⁸ field arithmetic + GF(2⁸) helpers.

Public API is authored in `f128` and re-exported here so
`from flock_zorch import field` / `field.<name>` call sites resolve unchanged.
"""
from flock_zorch.field.f128 import *  # noqa: F401,F403
from flock_zorch.field.f128 import _to_int, _to_lohi  # noqa: F401  (used cross-module by lincheck)
```

- [ ] **Step 3: Write `field/testing/__init__.py`**

```python
```
(empty file — package marker)

- [ ] **Step 4: Fix intra-group imports in the moved files**

In `field/gf8.py`: `from flock_zorch import _gf8_device` → `from flock_zorch.field import _gf8_device` (keep it at its current line 199 — it is a late import). Its `from flock_zorch import sumcheck` stays. `from flock_zorch import field, gf8` in `_gf8_device.py` → `from flock_zorch import field` + `from flock_zorch.field import gf8`. `_hostfield.py` has no internal imports.

- [ ] **Step 5: Rewrite every external reference to `gf8` / `_gf8_device` / `_hostfield`**

Find them:
```bash
grep -rn "from flock_zorch import" . | grep -E "\b(gf8|_gf8_device|_hostfield)\b"
```
Apply the master mapping to each (e.g. `zerocheck.py`'s `from flock_zorch import gf8, sumcheck` → `from flock_zorch import sumcheck` + `from flock_zorch.field import gf8`; `from flock_zorch import _hostfield as hf` → `from flock_zorch.field import _hostfield as hf`). `field` itself needs no rewrite (subpackage import + re-export).

- [ ] **Step 6: Verify no stale paths, gates green**

```bash
grep -rn "flock_zorch\.gf8\b\|flock_zorch\._gf8_device\|flock_zorch\._hostfield\|from flock_zorch import .*\bgf8\b" . ; echo "expect: no output"
cd /home/ryan/Workspace/flock-zorch2 && bazel test //python:all
```
Expected: grep empty; all PASS (`gf8_urm_oracle_test` now at `//python:gf8_urm_oracle_test`, still green).

- [ ] **Step 7: Commit**

```bash
git add -A python/flock_zorch
git commit -m "refactor(field): group F128/GF8 modules under field/ subpackage"
```

---

### Task 3: `hash/` subpackage

**Files:**
- Create: `flock_zorch/hash/__init__.py`, `flock_zorch/hash/testing/__init__.py`
- Move: `merkle.py`, `sha256.py` → `hash/`; `testing/{merkle_oracle_test,merkle_multi_oracle_test,merkle_openings_test,sha256_oracle_test}.py` → `hash/testing/`

- [ ] **Step 1: Create dirs + move**

```bash
cd /home/ryan/Workspace/flock-zorch2/python/flock_zorch
mkdir -p hash/testing
git mv merkle.py hash/merkle.py
git mv sha256.py hash/sha256.py
git mv testing/merkle_oracle_test.py hash/testing/
git mv testing/merkle_multi_oracle_test.py hash/testing/
git mv testing/merkle_openings_test.py hash/testing/
git mv testing/sha256_oracle_test.py hash/testing/
```

- [ ] **Step 2: Write `hash/__init__.py`**

```python
"""SHA-256 hash + SHA-256 Merkle commitment primitives."""
```
(leaf modules are imported directly as `from flock_zorch.hash import merkle` — no re-export needed.)

- [ ] **Step 3: Write `hash/testing/__init__.py`** (empty package marker)

```python
```

- [ ] **Step 4: Fix intra-group import in `hash/merkle.py`**

`from flock_zorch import sha256` → `from flock_zorch.hash import sha256`.

- [ ] **Step 5: Rewrite every external reference to `merkle` / `sha256`**

```bash
grep -rn "from flock_zorch import" . | grep -E "\b(merkle|sha256)\b"
```
Apply master mapping (e.g. `pcs_commit.py`'s `from flock_zorch import field, merkle` → `from flock_zorch import field` + `from flock_zorch.hash import merkle`).

- [ ] **Step 6: Verify + gates**

```bash
grep -rn "flock_zorch\.merkle\b\|flock_zorch\.sha256\b\|from flock_zorch import .*\bmerkle\b\|from flock_zorch import .*\bsha256\b" . ; echo "expect: no output"
cd /home/ryan/Workspace/flock-zorch2 && bazel test //python:all
```
Expected: grep empty; all PASS.
Note: `sha2_oracle_test`, `merkle_openings_test` are pure-host/CPU — covered by `//python:all`.

- [ ] **Step 7: Commit**

```bash
git add -A python/flock_zorch
git commit -m "refactor(hash): group merkle + sha256 under hash/ subpackage"
```

---

### Task 4: `sumcheck/` subpackage

**Files:**
- Create: `flock_zorch/sumcheck/__init__.py`, `flock_zorch/sumcheck/testing/__init__.py`
- Move: `sumcheck.py→sumcheck/eq.py`; `testing/{sumcheck_oracle_test,sumcheck_gpu_vs_cpu}.py`, `testing/bench_all.py` (imports only sumcheck) — **bench_all stays top-level** (it is the aggregate bench harness); move only the two sumcheck-specific ones.

- [ ] **Step 1: Create dirs + move**

```bash
cd /home/ryan/Workspace/flock-zorch2/python/flock_zorch
mkdir -p sumcheck/testing
git mv sumcheck.py sumcheck/eq.py
git mv testing/sumcheck_oracle_test.py sumcheck/testing/
git mv testing/sumcheck_gpu_vs_cpu.py sumcheck/testing/
```

- [ ] **Step 2: Write `sumcheck/__init__.py`**

```python
"""Sumcheck eq-table + fold multilinear utilities.

Authored in `eq`, re-exported so `from flock_zorch import sumcheck` /
`from flock_zorch.sumcheck import build_eq, ONE` resolve unchanged.
"""
from flock_zorch.sumcheck.eq import *  # noqa: F401,F403
```
(Exports `build_eq`, `build_eq_fused`, `ONE`, `fold_single`, `fold_pair`, `round_pair`, `eq_eval` — all non-underscore, covered by `*`.)

- [ ] **Step 3: Write `sumcheck/testing/__init__.py`** (empty)

```python
```

- [ ] **Step 4: `eq.py` has no intra-flock imports** — no source edit beyond the move. Verify with `grep -n "flock_zorch" sumcheck/eq.py` (expect none).

- [ ] **Step 5: External refs to `sumcheck` need no rewrite** (subpackage import + re-export). Confirm callers still say `from flock_zorch import sumcheck` or `from flock_zorch.sumcheck import build_eq…`:
```bash
grep -rn "flock_zorch.sumcheck\|from flock_zorch import sumcheck" .
```
No change required — the package + re-export preserve them.

- [ ] **Step 6: Gates**

```bash
cd /home/ryan/Workspace/flock-zorch2 && bazel test //python:all
```
Expected: all PASS (`sumcheck_oracle_test` at `//python:sumcheck_oracle_test`).

- [ ] **Step 7: Commit**

```bash
git add -A python/flock_zorch
git commit -m "refactor(sumcheck): move eq/fold utilities under sumcheck/ subpackage"
```

---

### Task 5: `zerocheck/` subpackage

**Files:**
- Create: `flock_zorch/zerocheck/__init__.py`, `flock_zorch/zerocheck/testing/__init__.py`
- Move: `zerocheck.py→zerocheck/prover.py`, `_zerocheck_fold.py→zerocheck/_fold.py`, `testing/zerocheck_oracle_test.py→zerocheck/testing/`

- [ ] **Step 1: Create dirs + move**

```bash
cd /home/ryan/Workspace/flock-zorch2/python/flock_zorch
mkdir -p zerocheck/testing
git mv zerocheck.py zerocheck/prover.py
git mv _zerocheck_fold.py zerocheck/_fold.py
git mv testing/zerocheck_oracle_test.py zerocheck/testing/
```

- [ ] **Step 2: Write `zerocheck/__init__.py`**

```python
"""Zerocheck sub-protocol (prover side).

Authored in `prover`, re-exported so `from flock_zorch import zerocheck` and
`from flock_zorch.zerocheck import _lagrange_weights` resolve unchanged.
"""
from flock_zorch.zerocheck.prover import *  # noqa: F401,F403
from flock_zorch.zerocheck.prover import _lagrange_weights  # noqa: F401  (used by lincheck)
```

- [ ] **Step 3: Write `zerocheck/testing/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Fix imports inside `zerocheck/prover.py` and `zerocheck/_fold.py`**

In `prover.py`: `from flock_zorch import _zerocheck_fold` → `from flock_zorch.zerocheck import _fold as _zerocheck_fold` (alias preserves call sites), OR direct `from flock_zorch.zerocheck._fold import <names>` matching the original form. `from flock_zorch import gf8, sumcheck` → `from flock_zorch import sumcheck` + `from flock_zorch.field import gf8`. `from flock_zorch import _hostfield as hf` → `from flock_zorch.field import _hostfield as hf`. `from flock_zorch import field` stays.
In `_fold.py`: `from flock_zorch import field` stays; `from flock_zorch import gf8`/`_hostfield` (if present) → field-qualified per master mapping.

- [ ] **Step 5: Rewrite external refs to `_zerocheck_fold`** (none expected outside the group — confirm):
```bash
grep -rn "flock_zorch\._zerocheck_fold\|from flock_zorch import _zerocheck_fold" . ; echo "expect: only inside zerocheck/"
```
`zerocheck` itself needs no external rewrite (subpackage + re-export).

- [ ] **Step 6: Gates**

```bash
cd /home/ryan/Workspace/flock-zorch2 && bazel test //python:all
```
Expected: all PASS (`zerocheck_oracle_test` green).

- [ ] **Step 7: Commit**

```bash
git add -A python/flock_zorch
git commit -m "refactor(zerocheck): group prover + fold under zerocheck/ subpackage"
```

---

### Task 6: `lincheck/` subpackage

**Files:**
- Create: `flock_zorch/lincheck/__init__.py`, `flock_zorch/lincheck/testing/__init__.py`
- Move: `lincheck.py→lincheck/prover.py`, `chain.py→lincheck/chain.py`, `keccak_lincheck.py→lincheck/keccak.py`, `keccak3_lincheck.py→lincheck/keccak3.py`, `_csc_fold.py→lincheck/_csc_fold.py`, `testing/{lincheck_oracle_test,lincheck_circuit_protocol_test,chain_shift_oracle_test}.py→lincheck/testing/`

- [ ] **Step 1: Create dirs + move**

```bash
cd /home/ryan/Workspace/flock-zorch2/python/flock_zorch
mkdir -p lincheck/testing
git mv lincheck.py lincheck/prover.py
git mv chain.py lincheck/chain.py
git mv keccak_lincheck.py lincheck/keccak.py
git mv keccak3_lincheck.py lincheck/keccak3.py
git mv _csc_fold.py lincheck/_csc_fold.py
git mv testing/lincheck_oracle_test.py lincheck/testing/
git mv testing/lincheck_circuit_protocol_test.py lincheck/testing/
git mv testing/chain_shift_oracle_test.py lincheck/testing/
```

- [ ] **Step 2: Write `lincheck/__init__.py`**

```python
"""Lincheck sub-protocol (prover) + hash-circuit seams.

Authored in `prover`, re-exported so `from flock_zorch import lincheck`,
`from flock_zorch.lincheck import CscCircuit, LincheckCircuit`, and
`from flock_zorch.lincheck import _round_eval, _bind_top` resolve unchanged.
"""
from flock_zorch.lincheck.prover import *  # noqa: F401,F403
from flock_zorch.lincheck.prover import _round_eval, _bind_top  # noqa: F401  (used by chain)
```
(`CscCircuit`, `LincheckCircuit`, `prove` are non-underscore → covered by `*`.)

- [ ] **Step 3: Write `lincheck/testing/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Fix intra-group imports**

- `lincheck/prover.py`: `from flock_zorch._csc_fold import _flatten_nz, _csc_segments, _seg_xor_fold` → `from flock_zorch.lincheck._csc_fold import _flatten_nz, _csc_segments, _seg_xor_fold`. `from flock_zorch.sumcheck import build_eq_fused, ONE` stays. `from flock_zorch.zerocheck import _lagrange_weights` stays (zerocheck re-exports it). `from flock_zorch.field import _to_int, _to_lohi` stays. `from flock_zorch import field` / `.challenger` stay.
- `lincheck/chain.py`: `from flock_zorch.lincheck import _round_eval, _bind_top` → `from flock_zorch.lincheck.prover import _round_eval, _bind_top` (direct submodule). `from flock_zorch.sumcheck import build_eq` stays. `from flock_zorch import field` stays.
- `lincheck/keccak3.py`: its `from flock_zorch import keccak_lincheck` → `from flock_zorch.lincheck.keccak import <names>` (direct; adjust to the exact symbols it used).
- `lincheck/keccak.py`: no internal flock imports (confirm).

- [ ] **Step 5: Rewrite external refs to `chain` / `keccak_lincheck` / `keccak3_lincheck`**

```bash
grep -rn "from flock_zorch import" . | grep -E "\b(chain|keccak_lincheck|keccak3_lincheck)\b"
```
Apply master mapping (`chain`→`from flock_zorch.lincheck import chain`; `keccak_lincheck`→`from flock_zorch.lincheck import keccak as keccak_lincheck`; `keccak3_lincheck`→`from flock_zorch.lincheck import keccak3 as keccak3_lincheck`). `lincheck` itself needs no external rewrite.

- [ ] **Step 6: Verify + gates**

```bash
grep -rn "flock_zorch\.chain\b\|flock_zorch\.keccak_lincheck\|flock_zorch\.keccak3_lincheck\|from flock_zorch import .*\bchain\b" . ; echo "expect: no output"
cd /home/ryan/Workspace/flock-zorch2 && bazel test //python:all
```
Expected: grep empty; all PASS (`lincheck_oracle_test`, `lincheck_circuit_protocol_test`, `chain_shift_oracle_test` green). Note: `keccak*` e2e tests are `_HEAVY` (venv) — deferred to Task 8.

- [ ] **Step 7: Commit**

```bash
git add -A python/flock_zorch
git commit -m "refactor(lincheck): group prover + chain + keccak circuits under lincheck/ subpackage"
```

---

### Task 7: `pcs/` subpackage

**Files:**
- Create: `flock_zorch/pcs/__init__.py`, `flock_zorch/pcs/testing/__init__.py`
- Move: `pcs.py→pcs/prover.py`, `pcs_commit.py→pcs/commit.py`, `pcs_open.py→pcs/open.py`, `basefold.py→pcs/basefold.py`, `fri.py→pcs/fri.py`, `ring_switch.py→pcs/ring_switch.py`, `zorch_ligerito.py→pcs/ligerito.py`, and the 11 pcs tests → `pcs/testing/`

- [ ] **Step 1: Create dirs + move**

```bash
cd /home/ryan/Workspace/flock-zorch2/python/flock_zorch
mkdir -p pcs/testing
git mv pcs.py pcs/prover.py
git mv pcs_commit.py pcs/commit.py
git mv pcs_open.py pcs/open.py
git mv basefold.py pcs/basefold.py
git mv fri.py pcs/fri.py
git mv ring_switch.py pcs/ring_switch.py
git mv zorch_ligerito.py pcs/ligerito.py
for t in commit pcs_open pcs_seam basefold basefold_verify ring_switch row_batch ligerito zorch_ligerito_driver zorch_ligerito_fs coding; do
  git mv testing/${t}_oracle_test.py pcs/testing/${t}_oracle_test.py
done
```

- [ ] **Step 2: Write `pcs/__init__.py`**

```python
"""flock PCS (Ligerito / BaseFold over F₂¹²⁸) in zorch's PcsProver seam shape.

The prover seam is authored in `prover`; `commit`/`open` are its phases.
Re-exported so `from flock_zorch.pcs import FlockPcsProver` resolves unchanged.
"""
from flock_zorch.pcs.prover import FlockPcsProver, FlockPcsProverData  # noqa: F401
```

- [ ] **Step 3: Write `pcs/testing/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Fix intra-group imports in moved pcs files**

- `pcs/prover.py`: `from flock_zorch import field, pcs_commit, pcs_open` → `from flock_zorch import field` + `from flock_zorch.pcs import commit as pcs_commit, open as pcs_open`. `from flock_zorch.challenger import Challenger` stays.
- `pcs/open.py`: `from flock_zorch.fri import default_fri_queries` → `from flock_zorch.pcs.fri import default_fri_queries`. Its **local** import inside `open()` `from flock_zorch import ring_switch, basefold` → `from flock_zorch.pcs import ring_switch, basefold` (keep it local/lazy).
- `pcs/commit.py`: `from flock_zorch import field, merkle` → `from flock_zorch import field` + `from flock_zorch.hash import merkle`. `from zorch.coding… import AdditiveReedSolomon` stays.
- `pcs/basefold.py`: `from flock_zorch import field, sumcheck, merkle, fri` → `from flock_zorch import field, sumcheck` + `from flock_zorch.hash import merkle` + `from flock_zorch.pcs import fri`. `from flock_zorch import _hostfield as hf` → `from flock_zorch.field import _hostfield as hf`.
- `pcs/fri.py`: `from flock_zorch import field` stays.
- `pcs/ring_switch.py`: `from flock_zorch import field, sumcheck` stays; `from flock_zorch.challenger import …` stays.
- `pcs/ligerito.py`: `from flock_zorch import field, merkle` → `from flock_zorch import field` + `from flock_zorch.hash import merkle`. `from zorch.… import …` stay.

- [ ] **Step 5: Rewrite external refs to all pcs leaves**

```bash
grep -rn "from flock_zorch import" . | grep -E "\b(pcs_commit|pcs_open|basefold|fri|ring_switch|zorch_ligerito)\b"
grep -rn "flock_zorch import pcs\b\|from flock_zorch\.pcs import" .
```
Apply master mapping across source + top-level `testing/` (e.g. `prover.py` top-level: `from flock_zorch import field, ring_switch, basefold, fri, zerocheck, lincheck, zorch_ligerito` → `from flock_zorch import field, zerocheck, lincheck` + `from flock_zorch.pcs import ring_switch, basefold, fri, ligerito as zorch_ligerito`; and `from flock_zorch.pcs import FlockPcsProver` stays).

- [ ] **Step 6: Verify no stale paths + CPU gates**

```bash
grep -rn "flock_zorch\.\(pcs_commit\|pcs_open\|basefold\|fri\|ring_switch\|zorch_ligerito\)\b\|from flock_zorch import .*\b\(pcs_commit\|pcs_open\|basefold\|fri\|ring_switch\|zorch_ligerito\)\b" . ; echo "expect: no output"
cd /home/ryan/Workspace/flock-zorch2 && bazel test //python:all
```
Expected: grep empty; all PASS (`pcs_open_oracle_test`, `pcs_seam_oracle_test`, `basefold_oracle_test`, `basefold_verify_oracle_test`, `ring_switch_oracle_test`, `row_batch_oracle_test`, `ligerito_oracle_test`, `coding_oracle_test` green). `commit`, `zorch_ligerito_driver` are `_HEAVY` (venv) — Task 8.

- [ ] **Step 7: Commit**

```bash
git add -A python/flock_zorch
git commit -m "refactor(pcs): group PCS prover/commit/open + basefold/fri/ring_switch/ligerito under pcs/ subpackage"
```

---

### Task 8: Final sweep — stale-path scan, docs/scripts, heavy/GPU gates

**Files:**
- Modify (if referenced): `python/flock_zorch/testing/run_commit_gates.sh`, `docs/SETUP.md`, `README.md`, `CLAUDE.md`

- [ ] **Step 1: Full stale-path scan across the repo**

```bash
cd /home/ryan/Workspace/flock-zorch2
grep -rn "flock_zorch\.\(field\|gf8\|merkle\|sha256\|zerocheck\|lincheck\|chain\|keccak_lincheck\|keccak3_lincheck\|pcs_commit\|pcs_open\|basefold\|fri\|ring_switch\|zorch_ligerito\)\b" python \
  | grep -vE "flock_zorch\.(field|hash|zerocheck|lincheck|pcs|sumcheck)\." ; echo "expect: no output"
```
Expected: no output. Any hit is a missed rewrite — fix it and re-run the relevant group's gate.

- [ ] **Step 2: Repoint `run_commit_gates.sh` + docs**

```bash
grep -rn "flock_zorch/testing/\(commit\|keccak\|sha2\|blake3\|zorch_ligerito_driver\)" python/flock_zorch/testing/run_commit_gates.sh docs/SETUP.md README.md CLAUDE.md
```
Update any path to its new home (commit gate → `flock_zorch/pcs/testing/commit_oracle_test.py`; keccak/sha2/blake3 e2e tests stay in the top-level `flock_zorch/testing/`). Fix only real references found.

- [ ] **Step 3: Run the heavy/GPU byte-identity gates on the venv**

Per `docs/SETUP.md`, run the `_HEAVY` gates (`commit`, `keccak`, `keccak_ligerito`, `keccak_chain`, `keccak3_ligerito`, `sha2`, `sha2_ligerito`, `blake3`, `blake3_ligerito`, `zorch_ligerito_driver`). Expected: all byte-match (green) — these are the definitive proof the reorg changed no bytes.

- [ ] **Step 4: Final full CPU gate**

```bash
cd /home/ryan/Workspace/flock-zorch2 && bazel test //python:all
```
Expected: all PASS.

- [ ] **Step 5: Self-review the diff**

```bash
git diff main --stat
git log --oneline main..HEAD
```
Confirm: every source diff is an import line or a file move (`git diff main -- '*.py' | grep '^[+-]' | grep -v '^[+-].*import' | grep -vE '^(\+\+\+|---)'` should show only docstring/`__init__` additions, no logic changes).

- [ ] **Step 6: Commit docs/script fixes**

```bash
git add -A
git commit -m "refactor(flock_zorch): repoint gate script + docs to subpackage layout"
```

---

## Self-Review (author checklist — completed)

- **Spec coverage:** field/hash/sumcheck/zerocheck/lincheck/pcs groups (Tasks 2–7) ✓; top-level `prover.py`/`challenger.py` unchanged ✓; per-group `testing/` ✓; single recursive-glob BUILD (Task 1) ✓; ~122-site import rewrite (master table, applied per task) ✓; external touch-ups (Task 8) ✓; staged bottom-up verification with gates ✓; heavy/GPU gates (Task 8) ✓.
- **Placeholder scan:** `__init__.py` contents, `git mv` commands, gate commands, and the rewrite mapping are all concrete. The per-line Edits are governed by the deterministic master mapping + a post-grep that must return empty — no "fix appropriately" latitude.
- **Type/name consistency:** renamed modules (`f128`, `eq`, `prover`×3, `commit`, `open`, `keccak`, `keccak3`, `ligerito`, `_fold`) and re-exported symbols (`_to_int`, `_to_lohi`, `_lagrange_weights`, `_round_eval`, `_bind_top`, `CscCircuit`, `LincheckCircuit`, `FlockPcsProver`, `FlockPcsProverData`) match across the mapping table, `__init__` files, and task steps.
- **Order:** bottom-up (field→hash→sumcheck→zerocheck→lincheck→pcs) guarantees each group's dependencies are already at their new paths before dependents move.
