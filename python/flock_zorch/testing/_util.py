# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared timing helper for the bench scripts.

Factored out of the per-bench copies so the timing methodology (warmup-excluded
best-of-n) lives in one place.
"""
from __future__ import annotations

import dataclasses
import time

import frx


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
