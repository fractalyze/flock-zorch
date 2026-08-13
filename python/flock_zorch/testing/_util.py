# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the suites under `**/testing/`: the random-ghash draw the
native tests use, and the timing methodology the bench scripts use.

Each is factored out of per-file copies so the convention lives in one place —
what "a random ghash" means, and warmup-excluded best-of-n timing.
"""
from __future__ import annotations

import dataclasses
import time

import frx
import frx.numpy as fnp
import numpy as np

from flock_zorch import ghash
from flock_zorch.testing._ptxas import (
    clmad_nvlink_verdict,
    clmad_ptxas_verdict,
    nvlink_version_text,
    ptxas_version_text,
)

_TOOLCHAIN_FIX = (
    "  Put ONE CUDA 13.3+ toolchain first on PATH — that covers both tools:\n"
    "      CUDA13=/path/to/cuda-13.3\n"
    '      export PATH="$CUDA13/bin:$PATH"\n'
    "  Keep those two lines separate. In a single\n"
    '      export CUDA_ROOT=<root> PATH="$CUDA_ROOT/bin:$PATH"\n'
    "  the $CUDA_ROOT on the right still expands to its OLD value, so the "
    "toolchain never reaches PATH — which is how this is usually hit.\n"
    "  If a mis-toolchained run already populated JAX_COMPILATION_CACHE_DIR, "
    "wipe it: the toolchain is excluded from the cache key, so its "
    "executables stay hits."
)


def gate_device():
    """The device a byte gate runs on, refusing a toolchain it cannot trust.

    `prove_phase_bench` guards its measurements because a pre-13.3 ptxas
    silently costs ~15x. A byte gate is guarded for a sharper reason: a wrong
    toolchain does not always degrade quietly, and when it fails it does so
    where the gate reports bytes. Two ways it bites, both measured here:

    - ptxas below 13.3 drops clmad, so the GF(2^128) multiply goes shift/XOR.
    - ptxas and nvlink disagree, because XLA takes them from different places
      (`xla_gpu_cuda_data_dir` vs PATH then `/usr/local/cuda`). Exporting only
      `CUDA_ROOT` assembles with 13.3 and links with a 12.9 nvlink; on the m30
      Ligerito gate that dies with `nvlink fatal: Internal FNLZR error`.

    The second cost a filed issue and a session, diagnosed as a prover
    regression in the PR that happened to merge before it, so the gates refuse
    the toolchain up front rather than let it surface as bytes. Like the
    bench's guard, an unreadable probe on either side lets the run proceed
    rather than blocking on an ambiguous signal.
    """
    device = frx.devices()[0]
    print(f"device {device}")
    if device.platform != "gpu":
        return device
    cc = getattr(device, "compute_capability", None)
    reason = clmad_ptxas_verdict(ptxas_version_text(), cc) or clmad_nvlink_verdict(
        nvlink_version_text(), cc
    )
    if reason:
        raise SystemExit(f"REFUSING to gate: {reason}.\n{_TOOLCHAIN_FIX}")
    return device


def rand_ghash(rng: np.random.Generator, n: int):
    """`binary_field_ghash [n]` drawn from `rng`, uniform over GF(2^128).

    The lane bound is the full `2**64` — `Generator.integers` accepts it as the
    exclusive high for `dtype=uint64` — so every bit of both limbs is uniform
    and the draw covers the whole field. The per-file copies this consolidates
    had drifted between this and `1 << 63`, which always cleared the top bit of
    each limb; no recorded reason for the narrower bound survives (the dtype is
    uint64 end-to-end, so int64 signedness never enters), and the full-bound
    draw was already in use elsewhere in the suite.
    """
    lanes = rng.integers(0, 2**64, size=(n, 2), dtype=np.uint64)
    return ghash.to_ghash(fnp.asarray(lanes))


def deep_leaves(x):
    """`tree_leaves(x)`, but descending into dataclasses that are not pytrees.

    `tree_leaves` stops at a dataclass nobody registered as a pytree node and
    yields the object ITSELF as one opaque leaf, so its arrays never reach
    `block_until_ready`. flock's claims and proofs (`ZerocheckClaim`,
    `ZerocheckProof`, ...) are exactly that shape, which made `await_all` a
    silent no-op on them. Non-array leaves ride along as `tree_leaves` returns
    them; `block_until_ready` passes anything it cannot block on through.
    """
    leaves = []
    for leaf in frx.tree_util.tree_leaves(x):
        if dataclasses.is_dataclass(leaf) and not isinstance(leaf, type):
            fields = [getattr(leaf, f.name) for f in dataclasses.fields(leaf)]
            leaves.extend(deep_leaves(fields))
        else:
            leaves.append(leaf)
    return leaves


def await_all(x):
    """Block until every frx leaf of `x` is materialized, so async dispatch
    cannot leak past a timing boundary."""
    frx.block_until_ready(deep_leaves(x))
    return x


def best_of(fn, n=3):
    """Warmup-excluded best-of-`n` wall-clock ms, keeping the fastest run's own
    breakdown. `fn` returns `(result, detail)`; the returned `detail` is the one
    belonging to the run whose time is reported, so a caller that times sub-steps
    never mixes a total from one run with a split from another.

    One warmup call (its compile/first transfer excluded) precedes the timed
    runs; `min` discards scheduler jitter.
    """
    await_all(fn()[0])
    best_ms, best_detail = float("inf"), None
    for _ in range(n):
        t0 = time.perf_counter()
        result, detail = fn()
        await_all(result)
        ms = (time.perf_counter() - t0) * 1e3
        if ms < best_ms:
            best_ms, best_detail = ms, detail
    return best_ms, best_detail


def best(fn, n=3):
    """Best-of-`n` wall-clock ms for `fn()` — `best_of` with no per-run detail."""
    return best_of(lambda: (fn(), None), n)[0]


def report(results, summary: str) -> int:
    """Print one PASS/FAIL line per `(name, ok)` plus a summary, and return the
    process exit code. Every byte gate ends this way."""
    allok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{summary}: {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1
