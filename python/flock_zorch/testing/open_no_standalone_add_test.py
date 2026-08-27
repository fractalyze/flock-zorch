"""No full-size elementwise op may be dispatched as its own XLA module.

A module holding one instruction has nothing to fuse against, so it reads its
inputs from DRAM and writes its output back for work that belongs inside a
neighbour. At m32 the batched claim combine did exactly this: an XOR-sum over a
2^25 ghash array cost 1.465 GiB of round-trip.

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
        bad = _offending_modules(dump_dir)
        return [
            (
                f"no full-size standalone module (found: {bad or 'none'})",
                bad == [],
            )
        ]
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(report(run(), "open emits no full-size standalone module"))
