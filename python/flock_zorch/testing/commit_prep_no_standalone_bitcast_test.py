# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The commit path's witness bitcast must not dispatch as its own module.

`_commit_prep` (`pcs/ligerito.py`) traces the witness bitcast and the
bit-reverse together for exactly one reason: a bitcast dispatched alone is
not free. Its module gets root `copy(bitcast(param))` — the copy exists only
to give the module a distinct output buffer, and it is a real memcpy over the
whole witness (512 MiB at m32). Traced with the permutation that follows it,
the bitcast disappears into the bit-reverse's fused kernel instead.

`commit_prep_test.py`, next to this file, only checks that `_commit_prep`
computes the right bytes. That is not enough to guard the fix: calling the
eager `_bitrev(ghash.to_ghash(z))` at the call site instead of `_commit_prep`
computes the SAME bytes through two DRAM round trips, so a numeric-only test
is blind to the regression this file exists to catch. This test dumps the
commit window's compiled HLO and asserts no `jit_bitcast_convert_type` module
carrying the full witness's own shape roots a `copy` of a `bitcast`.

Verified both directions by hand before writing this test: on the fix
(`52216bd`), no `jit_bitcast_convert_type` module carries the witness shape —
the ingest step does emit a handful of scalar/small-array modules under that
same primitive name (sizes 1, 3, 4, 8, 256), which is `KNOWN_OFFENDERS`'
`jit_bitcast_convert_type` note in `open_no_standalone_add_test.py` in a
different form: the primitive name alone does not identify the call site, so
this test keys on the witness's own shape, not the name. Reverting
`commit_flock_ligerito`'s call site to `prover.commit([_bitrev(ghash.to_ghash(z))])`
reproduces `module_0017.jit_bitcast_convert_type`, entry layout
`u64[32768,2]{1,0}->binary_field_ghash[32768]{0}`, root
`%copy.1 = binary_field_ghash[32768]{0} copy(%bitcast.1)` — exactly the shape
and pattern this test checks for.

Structural rather than numerical on purpose, and dumped from the smallest
available golden (`blake3_ligerito_golden.bin`, m22) rather than m32 — the
copy-of-bitcast pattern depends on whether the bitcast runs inside a trace or
alone, not on witness size, so the cheapest golden that reproduces it is
enough; the hand-check above confirms m22 does.
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

from flock_zorch.testing._util import report

# u64[32768, 2] -> binary_field_ghash[32768]: the L0 commit witness shape at
# the default (m22) golden, after `commit_flock_ligerito` reshapes it to
# (N, 2) lanes. A `jit_bitcast_convert_type` module whose output carries this
# exact shape is operating on the whole witness, not one of the small
# ingest-time scalar/config conversions that share the primitive name.
WITNESS_SHAPE = "binary_field_ghash[32768]"


def _dump_commit_hlo(dump_dir):
    env = dict(os.environ, XLA_FLAGS=f"--xla_dump_to={dump_dir}")
    return subprocess.run(
        [
            sys.executable,
            "python/flock_zorch/testing/nsys_capture.py",
            "--window",
            "commit",
            "--golden",
            "blake3_ligerito_golden.bin",
            "--walls",
            "1",
        ],
        env=env,
        capture_output=True,
        text=True,
    )


def _witness_bitcast_copies(dump_dir):
    """Names of `jit_bitcast_convert_type` modules that both carry the full
    witness shape and root a `copy` of a `bitcast` — the eager-dispatch
    signature the fix removed."""
    bad = []
    for path in glob.glob(os.path.join(dump_dir, "*after_optimizations.txt")):
        name = os.path.basename(path).split(".")[1]
        if name != "jit_bitcast_convert_type":
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if not lines or WITNESS_SHAPE not in lines[0]:
            continue
        root = next((ln for ln in lines if "ROOT" in ln), "")
        if "copy(" in root and "bitcast" in root:
            bad.append(name)
    return sorted(bad)


def run():
    dump_dir = tempfile.mkdtemp(prefix="hlo_commit_prep_gate_")
    try:
        proc = _dump_commit_hlo(dump_dir)
        if proc.returncode != 0:
            return [("commit window compiled", False)]
        bad = _witness_bitcast_copies(dump_dir)
        return [
            (
                f"no standalone copy(bitcast(...)) of the full witness "
                f"(found: {bad or 'none'})",
                bad == [],
            )
        ]
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(report(run(), "commit prep emits no standalone witness bitcast copy"))
