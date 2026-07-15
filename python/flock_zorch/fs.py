"""Jitted Fiat-Shamir hops over the device transcript.

An eager transcript op dispatches each of its ~10 internal primitives
separately (~40-190 ms per op on the CPU backend, similar dispatch overhead on
GPU), so a protocol making hundreds of them spends minutes on Fiat-Shamir
alone. Every hop here is one compiled executable; inputs are scalar- or
static-shaped, so each compiles once per distinct shape (labels compile once
per literal) and is cached for the process.
"""
from __future__ import annotations

import functools

import frx


@frx.jit
def observe_scalar(t, x):
    """Scalar-framed observe — `x` 0-d for one op, `[n]` for n ops."""
    return t.observe_scalar(x)


@frx.jit
def observe_slice(t, xs):
    return t.observe(xs)


@frx.jit
def observe_bytes(t, data):
    return t.observe_bytes(data)


@functools.partial(frx.jit, static_argnums=(1,))
def observe_label(t, label):
    return t.observe_label(label)


@frx.jit
def sample_scalar(t):
    return t.sample_scalar()


@functools.partial(frx.jit, static_argnums=(1,))
def sample_slice(t, n):
    return t.sample(n)


@functools.partial(frx.jit, static_argnums=(1,))
def grind(t, bits):
    """Device PoW grind — the transcript's windowed search as one program."""
    return t.grind(bits)


@functools.partial(frx.jit, static_argnums=(2,))
def check_witness(t, witness, bits):
    return t.check_witness(witness, bits)


@frx.jit
def observe_pair_sample(t, x, y):
    """observe_scalar(x) → observe_scalar(y) → sample_scalar(): the
    (message, challenge) hop every sumcheck round makes, as one device program."""
    t = t.observe_scalar(x).observe_scalar(y)
    return t.sample_scalar()


@functools.partial(frx.jit, static_argnums=(1,))
def sample_chain(t, n):
    """n successive scalar-framed squeezes as one device program. Each squeeze
    re-absorbs, so the chain is sequential — but shape-uniform per step, so it
    scans (O(1) compile) rather than unrolling."""
    return frx.lax.scan(lambda t, _: t.sample_scalar(), t, None, length=n)
