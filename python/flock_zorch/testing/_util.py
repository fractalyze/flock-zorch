# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared timing + field-mul-backend helpers for the bench scripts.

Factored out of the per-bench copies so the timing methodology (warmup-excluded
best-of-n) lives in one place. Not used by the `*_oracle_test.py` byte-match gates
— those select their mul inline so the gate stays self-contained.
"""
from __future__ import annotations

import time

import jax

from flock_zorch import field


def select_mul():
    """The software GF(2^128) multiply `field.mul`."""
    return field.mul


def best(fn, n=3):
    """Best-of-`n` wall-clock ms for `fn()`. One warmup call (its compile/transfer
    excluded) precedes the timed runs; every jax output leaf is awaited so async
    dispatch can't leak into the measurement. `min` discards scheduler jitter."""
    r = fn()
    jax.block_until_ready(jax.tree_util.tree_leaves(r))
    b = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        jax.block_until_ready(jax.tree_util.tree_leaves(r))
        b = min(b, time.perf_counter() - t0)
    return b * 1e3
