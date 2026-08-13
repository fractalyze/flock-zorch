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
# `<major>.<minor>` out of a `ptxas`/`nvlink` --version banner, or None.
_RELEASE_RE = re.compile(r"release (\d+)\.(\d+)")
# First compute capability with clmad — Blackwell sm_120a. An older card has no
# fast path at ANY toolchain, so there the fallback is the measurement and the
# guard stays out of the way.
_CLMAD_CC_FLOOR = 12


def _release(version_text: str | None) -> tuple[int, int] | None:
    """`(major, minor)` out of a CUDA tool's `--version` banner, or None."""
    m = _RELEASE_RE.search(version_text) if version_text else None
    return (int(m.group(1)), int(m.group(2))) if m else None


# XLA's fallback CUDA root when nothing else resolves a tool.
_XLA_DEFAULT_CUDA_ROOT = "/usr/local/cuda"


def _in_bin(root: str, tool: str) -> str | None:
    """`<root>/bin/<tool>` if it is executable, else None."""
    candidate = os.path.join(root, "bin", tool) if root else ""
    return candidate if candidate and os.access(candidate, os.X_OK) else None


def _version_text(candidates: tuple[str | None, ...]) -> str | None:
    """`--version` of the first candidate that resolves, or None."""
    path = next((c for c in candidates if c), None)
    if path is None:
        return None
    try:
        return subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except Exception:
        return None


def clmad_nvlink_verdict(
    version_text: str | None, compute_capability: str | None
) -> str | None:
    """The reason a run would link clmad cubins with too old an nvlink, or None.

    XLA resolves the assembler and the linker from DIFFERENT places, so they
    can disagree. `ptxas` follows `xla_gpu_cuda_data_dir`, which frx sets from
    an exported `CUDA_ROOT`; the linker lookup does not use that directory and
    falls back to PATH then `/usr/local/cuda`. Exporting only `CUDA_ROOT`
    therefore assembles PTX 9.3 with a 13.3 ptxas and hands the result to a
    12.9 nvlink, which dies with `nvlink fatal: Internal FNLZR error` — on the
    m30 Ligerito gate that read as a byte regression in whatever merged last.

    Checking the linker against the same floor rather than against the ptxas
    probe is deliberate: which ptxas XLA picks is genuinely ambiguous from the
    environment (frx defaults `CUDA_ROOT` when unset, so reading it back cannot
    tell a user's export from frx's), while the linker is not. One 13.3+
    toolchain first on PATH satisfies both, which is what the README says.

    Same conservatism as the ptxas verdict: an unreadable probe, or a card
    with no clmad at any toolchain, returns None rather than blocking.
    """
    if not version_text or not compute_capability:
        return None
    version = _release(version_text)
    cc = re.match(r"(\d+)", compute_capability)
    if not version or not cc or int(cc.group(1)) < _CLMAD_CC_FLOOR:
        return None
    if version >= CLMAD_PTXAS_FLOOR:
        return None
    return (
        f"nvlink {version[0]}.{version[1]} is older than the "
        f"{CLMAD_PTXAS_FLOOR[0]}.{CLMAD_PTXAS_FLOOR[1]} that assembles clmad, "
        "so it may fail to link the cubins it is handed"
    )


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
    version = _release(version_text)
    cc = re.match(r"(\d+)", compute_capability)
    if not version or not cc or int(cc.group(1)) < _CLMAD_CC_FLOOR:
        return None
    if version >= CLMAD_PTXAS_FLOOR:
        return None
    return (
        f"ptxas {version[0]}.{version[1]} cannot assemble clmad (PTX ISA 9.3 "
        f"needs {CLMAD_PTXAS_FLOOR[0]}.{CLMAD_PTXAS_FLOOR[1]}+), so the "
        "GF(2^128) multiply falls back to shift/XOR"
    )


def ptxas_version_text() -> str | None:
    """`ptxas --version` for the ptxas XLA will actually run, or None.

    XLA resolves ptxas from `CUDA_DIR`, then PATH — so that is the probe
    order. `CUDA_ROOT` is deliberately NOT read: frx's Mosaic GPU module
    exports `CUDA_ROOT=/usr/local/cuda` at import time as a libdevice hint
    (`frx/experimental/mosaic/gpu/core.py`), so by the time this runs it
    names a toolchain that plays no part in assembling the kernels — reading
    it falsely refused a correctly-PATH'd run whose proves demonstrably
    assembled clmad.
    """
    return _version_text(
        (_in_bin(os.environ.get("CUDA_DIR", ""), "ptxas"), shutil.which("ptxas"))
    )


def nvlink_version_text() -> str | None:
    """`nvlink --version` for the nvlink XLA will actually run, or None.

    PATH first, then XLA's `/usr/local/cuda` fallback. `CUDA_ROOT` is not
    consulted for the same reason as above, and `CUDA_DIR` is not either —
    measured, the linker lookup ignores that directory. With `CUDA_ROOT` at a
    13.3 tree and no nvlink anywhere on PATH, the m30 gate still linked with
    the 12.9 `/usr/local/cuda/bin/nvlink` and died there; probing only PATH
    would have returned None and waved that exact run through.
    """
    return _version_text(
        (shutil.which("nvlink"), _in_bin(_XLA_DEFAULT_CUDA_ROOT, "nvlink"))
    )
