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
