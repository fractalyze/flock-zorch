# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Dump a prover window's optimised HLO and walk the modules it emitted.

Structural gates assert on what the compiler emitted — which module exists, what
shape its entry carries, what its root is — because the effects they guard are
invisible to the numeric and wall-clock channels: an operation dispatched as its
own one-instruction module computes the same answer at the same speed the
harness can resolve, and only the module list shows it round-tripping DRAM for
work that belonged inside a neighbour.

`XLA_FLAGS` must be set before the XLA client initialises, so the dump runs in a
subprocess rather than in-process.
"""
from __future__ import annotations

import contextlib
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator

_CAPTURE = "python/flock_zorch/testing/nsys_capture.py"


@contextlib.contextmanager
def dump_window_hlo(window: str, golden: str) -> Iterator[str | None]:
    """Compile `window` once and yield the directory its HLO dump landed in, or
    `None` if the subprocess failed.

    `--walls 1` is not a timing run — a single un-warmed iteration reads several
    times the warm median. It is the cheapest way to force a compile.
    """
    dump_dir = tempfile.mkdtemp(prefix=f"hlo_{window}_")
    try:
        proc = subprocess.run(
            [
                sys.executable,
                _CAPTURE,
                "--window",
                window,
                "--golden",
                golden,
                "--walls",
                "1",
            ],
            env=dict(os.environ, XLA_FLAGS=f"--xla_dump_to={dump_dir}"),
            capture_output=True,
            text=True,
        )
        yield dump_dir if proc.returncode == 0 else None
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


def modules(dump_dir: str) -> Iterator[tuple[str, list[str]]]:
    """Yield `(module_name, lines)` for every optimised-HLO module in the dump.

    The name is the jitted function's, parsed out of
    `module_NNNN.<name>.<backend>_after_optimizations.txt`. It is **not**
    call-site unique: every eager dispatch of the same primitive anywhere in the
    tree lands under one name, so a gate that must distinguish two sites has to
    key on shape or on emitted structure as well.
    """
    for path in sorted(glob.glob(os.path.join(dump_dir, "*after_optimizations.txt"))):
        name = os.path.basename(path).split(".")[1]
        with open(path, encoding="utf-8", errors="replace") as f:
            yield name, f.readlines()
