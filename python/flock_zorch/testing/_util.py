# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared timing helper for the bench scripts.

Factored out of the per-bench copies so the timing methodology (warmup-excluded
best-of-n) lives in one place.
"""
from __future__ import annotations

import time

import frx


def await_all(x):
    """Block until every frx leaf of `x` is materialized, so async dispatch
    cannot leak past a timing boundary."""
    frx.block_until_ready(frx.tree_util.tree_leaves(x))
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
