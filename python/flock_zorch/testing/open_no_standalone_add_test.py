"""No full-size elementwise op may be dispatched as its own XLA module.

A module holding one instruction has nothing to fuse against, so it reads its
inputs from DRAM and writes its output back for work that belongs inside a
neighbour. At m32 the batched claim combine did exactly this: an XOR-sum over a
2^25 ghash array cost 1.465 GiB of round-trip. Moving the combine's call site
inside `_open_jitted`'s trace fixed that one instance (`COMBINE_MODULE`
below); this gate holds it fixed.

The open window emits five other full-size standalone modules
(`KNOWN_OFFENDERS` below) that are the same class of bug but out of scope for
the combine fix — separate, not-yet-fixed materialisations. This gate
tolerates them BY NAME rather than asserting only "no new one appeared": the
list shrinks when one of them is individually fixed (fused into its
consumer, the same way the combine was), and grows only if this test is
edited to add a name — never automatically. Do not add a name to
`KNOWN_OFFENDERS` to silence a newly-appeared offender; that is a
regression, not a known one, and belongs in `COMBINE_MODULE`'s class of
finding (something moved that shouldn't have).

Structural rather than numerical on purpose — the combine's arithmetic is
unchanged by where it runs, so only the emitted module list can catch a
regression here.
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

from flock_zorch.testing._util import report

# 2^25 ghash elements: the full-size witness/basis at m32. A module whose entry
# layout mentions this shape is operating on the whole polynomial.
FULL_SIZE = "binary_field_ghash[33554432]"

# Modules that legitimately carry a full-size operand. Anything else naming a
# single primitive is a materialisation that should have been traced with its
# consumer.
ALLOWED_PREFIXES = ("jit__open_jitted", "jit__commit", "jit__mlv_sumcheck")

# The regression this task guards against: the batched claim combine
# (`_combine_claims`), previously called outside any jit, materialised as this
# standalone XOR-sum module. If this name is emitted again, the combine call
# moved back out of `_open_jitted`'s trace.
COMBINE_MODULE = "jit_add"

# Full-size standalone modules present both before and after the combine fix,
# unrelated to `_combine_claims`. Same class of bug, separate (unfixed) task —
# named explicitly so the gate stays green today without masking a NEW
# offender. Remove a name only when that module has actually been fixed;
# never add one to make a fresh regression pass.
KNOWN_OFFENDERS = frozenset(
    {
        "jit__slice_evals",
        "jit_bit_reverse",
        "jit_bitcast_convert_type",
        "jit_build_eq_suffix_tables",
        "jit_rs_eq_ind",
    }
)


def _dump_open_hlo(dump_dir):
    env = dict(os.environ, XLA_FLAGS=f"--xla_dump_to={dump_dir}")
    return subprocess.run(
        [
            sys.executable,
            "python/flock_zorch/testing/nsys_capture.py",
            "--window",
            "open",
            "--golden",
            "blake3_ligerito_golden_m32.bin",
            "--walls",
            "1",
        ],
        env=env,
        capture_output=True,
        text=True,
    )


def _offending_modules(dump_dir):
    bad = []
    for path in glob.glob(os.path.join(dump_dir, "*after_optimizations.txt")):
        name = os.path.basename(path).split(".")[1]
        if name.startswith(ALLOWED_PREFIXES):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.readline()
        if FULL_SIZE in head and "entry_computation_layout" in head:
            bad.append(name)
    return sorted(bad)


def run():
    dump_dir = tempfile.mkdtemp(prefix="hlo_open_gate_")
    try:
        proc = _dump_open_hlo(dump_dir)
        if proc.returncode != 0:
            return [("open window compiled", False)]
        bad = set(_offending_modules(dump_dir))
        new = sorted(bad - KNOWN_OFFENDERS - {COMBINE_MODULE})
        return [
            (
                f"{COMBINE_MODULE} absent — combine still inside the open's " "trace",
                COMBINE_MODULE not in bad,
            ),
            (
                f"no new full-size standalone module beyond the known "
                f"{sorted(KNOWN_OFFENDERS)} (found: {new or 'none'})",
                new == [],
            ),
        ]
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(report(run(), "open emits no regressed / new full-size standalone module"))
