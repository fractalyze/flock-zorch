# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Whether the ptxas a CUDA run will assemble with can reach the clmad path.

On a clmad-capable card the GF(2^128) multiply lowers to `clmad.{lo,hi}.u64`,
which needs PTX ISA 9.3 — first assembled by the CUDA 13.3 ptxas. XLA derives
the PTX version it emits from the toolchain it finds, so an older ptxas does
not fail: the multiply silently falls back to shift/XOR and only the
clmul-heavy phases blow up (zerocheck ~17x, open ~7x at m32 — a ~15x wall that
reads as a mystery regression, not as an error). Worse, the run poisons the
per-wheel XLA compile cache with the slow executables, so later runs at the
right toolchain reload them. The bench refuses that state up front, the same
way it refuses a contended card.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

# First ptxas that assembles PTX ISA 9.3 (`clmad.{lo,hi}.u64`).
CLMAD_PTXAS_FLOOR = (13, 3)
# First compute capability with clmad — Blackwell sm_120a. An older card has no
# fast path at ANY toolchain, so there the fallback is the measurement and the
# guard stays out of the way.
_CLMAD_CC_FLOOR = 12


def clmad_ptxas_verdict(
    version_text: str | None, compute_capability: str | None
) -> str | None:
    """The reason a measurement would miss the clmad fast path, or None.

    Acts only on the one unambiguous fact, like the contention guard: a missing
    probe, unparseable output, or a card with no clmad at any toolchain all
    return None rather than block.
    """
    if not version_text or not compute_capability:
        return None
    release = re.search(r"release (\d+)\.(\d+)", version_text)
    cc = re.match(r"(\d+)", compute_capability)
    if not release or not cc or int(cc.group(1)) < _CLMAD_CC_FLOOR:
        return None
    version = (int(release.group(1)), int(release.group(2)))
    if version >= CLMAD_PTXAS_FLOOR:
        return None
    return (
        f"ptxas {version[0]}.{version[1]} cannot assemble clmad (PTX ISA 9.3 "
        f"needs {CLMAD_PTXAS_FLOOR[0]}.{CLMAD_PTXAS_FLOOR[1]}+), so the "
        "GF(2^128) multiply falls back to shift/XOR"
    )


def ptxas_version_text() -> str | None:
    """`ptxas --version` for the ptxas a run would pick up, or None.

    `$CUDA_ROOT/bin/ptxas` first — CUDA_ROOT, not PATH, is what selects the
    toolchain (README's CUDA-13.3 note) — then PATH as the fallback probe.
    """
    root = os.environ.get("CUDA_ROOT", "")
    candidate = os.path.join(root, "bin", "ptxas") if root else None
    ptxas = candidate if candidate and os.access(candidate, os.X_OK) else None
    ptxas = ptxas or shutil.which("ptxas")
    if ptxas is None:
        return None
    try:
        return subprocess.run(
            [ptxas, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except Exception:
        return None
