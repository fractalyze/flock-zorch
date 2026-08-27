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

The ingest step emits a handful of small modules (sizes 1, 3, 4, 8, 256) under
that same primitive name, so the name alone cannot say which call site produced
one — this gate keys on the witness's own shape as well.

Structural rather than numerical on purpose: the eager form computes the same
answer, and the only difference it makes is a module the numeric test cannot
see.
"""

import sys

from flock_zorch.testing import _hlo as hlo
from flock_zorch.testing._util import report

# The copy-of-bitcast pattern depends on whether the reinterpretation runs
# inside a trace or alone, not on witness size, so the standing gate's golden
# is enough and an m32 prove would cost minutes for the same evidence.
GOLDEN = "blake3_ligerito_golden.bin"
# u64[32768, 2] -> binary_field_ghash[32768]: the L0 commit witness shape at
# the default (m22) golden, after `commit_flock_ligerito` reshapes it to
# (N, 2) lanes. A `jit_bitcast_convert_type` module whose output carries this
# exact shape is operating on the whole witness, not one of the small
# ingest-time scalar/config conversions that share the primitive name.
WITNESS_SHAPE = "binary_field_ghash[32768]"


def _witness_bitcast_copies(dump_dir):
    """Names of `jit_bitcast_convert_type` modules that both carry the full
    witness shape and root a `copy` of a `bitcast` — the eager-dispatch
    signature the fix removed."""
    bad = []
    for name, lines in hlo.modules(dump_dir):
        if name != "jit_bitcast_convert_type":
            continue
        if not lines or WITNESS_SHAPE not in lines[0]:
            continue
        root = next((ln for ln in lines if "ROOT" in ln), "")
        if "copy(" in root and "bitcast" in root:
            bad.append(name)
    return sorted(bad)


def run():
    with hlo.dump_window_hlo("commit", GOLDEN) as dump_dir:
        if dump_dir is None:
            return [("commit window compiled", False)]
        bad = _witness_bitcast_copies(dump_dir)
        return [
            (
                f"no standalone copy(bitcast(...)) of the full witness "
                f"(found: {bad or 'none'})",
                bad == [],
            )
        ]


if __name__ == "__main__":
    sys.exit(report(run(), "commit prep emits no standalone witness bitcast copy"))
