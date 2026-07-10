"""Jitted Fiat-Shamir hops over the device transcript.

An eager transcript op is one XLA executable dispatch (~40-190 ms each on the
CPU backend), so a protocol that makes hundreds of them spends minutes on
Fiat-Shamir alone. These helpers fuse the recurring op sequences into single
compiled programs; every input is scalar-shaped, so each compiles once per
process regardless of instance size.
"""
from __future__ import annotations

import functools

import jax


@jax.jit
def observe_pair_sample(t, x, y):
    """observe_scalar(x) → observe_scalar(y) → sample_scalar(): the
    (message, challenge) hop every sumcheck round makes, as one device program."""
    t = t.observe_scalar(x).observe_scalar(y)
    return t.sample_scalar()


@functools.partial(jax.jit, static_argnums=(1,))
def sample_chain(t, n):
    """n successive scalar-framed squeezes as one device program. Each squeeze
    re-absorbs, so the chain is sequential — but shape-uniform per step, so it
    scans (O(1) compile) rather than unrolling."""
    return jax.lax.scan(lambda t, _: t.sample_scalar(), t, None, length=n)
