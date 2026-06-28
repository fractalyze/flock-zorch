"""`GhashFieldOps` — flock's GF(2^128) as zorch's `FieldOps` seam.

zorch added `zorch.sumcheck.field_ops.FieldOps` (a `Protocol`) so a field carried
as raw `uint64` lanes — exactly flock's GF(2^128) in the GHASH basis, where `+` is
XOR and `*` is a carryless GHASH multiply — can drive zorch's shared multilinear
sumcheck without a jax-native dtype. The commit was authored anticipating flock.
This module is flock's instantiation of that seam: the single typed surface for
flock's field arithmetic, faithful to `field.py` + `sumcheck._xor_reduce` (pinned
by `field_ops_test.py`, the same NativeFieldOps-parity discipline zorch uses).

Scope today is the SEAM, not a driver swap. flock keeps its own host round loop
(its Fiat-Shamir is a sequential host SHA-256 Merlin transcript) and its own
Karatsuba ∞-trick round message `(r·G(1), G(∞))` — which is NOT the degree+1-evals
domain form, so `domain_point` raises (per the protocol: "its round owns the
message"). `GhashFieldOps` is the on-ramp: it lets flock express field arithmetic
against the shared seam and is the first step toward reusing zorch's sumcheck /
the zkx GPU sumcheck emitter once a binary-field driver exists.

`mul` is parameterized (software `field.mul` vs the `field_clmad.mul` FFI), the
same selection flock threads as `mul=` elsewhere — so a `GhashFieldOps(mul)` bundles
that choice with the rest of the field surface in one object.

Requires `jax_enable_x64`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from flock_zorch import field

U64 = jnp.uint64
_ZERO = jnp.zeros((2,), U64)        # F128::ZERO = {lo: 0, hi: 0}
_ONE = jnp.asarray([1, 0], U64)     # F128::ONE  = {lo: 1, hi: 0}


class GhashFieldOps:
    """`zorch.sumcheck.field_ops.FieldOps` for flock's GF(2^128) (GHASH basis,
    uint64 [..., 2] lanes). `add`/`sub` are XOR (characteristic 2); `sum` is an
    XOR-reduce; `mul` is the GHASH product (`field.mul` or `field_clmad.mul`).

    Byte-identical to flock's bare primitives (`field.add`, `field.mul`,
    `sumcheck._xor_reduce`), pinned by `field_ops_test.py`."""

    def __init__(self, mul=field.mul):
        self._mul = mul

    @property
    def zero(self) -> Array:
        return _ZERO

    @property
    def one(self) -> Array:
        return _ONE

    def add(self, a: Array, b: Array) -> Array:
        return a ^ b

    def sub(self, a: Array, b: Array) -> Array:
        return a ^ b  # characteristic 2: subtraction is addition

    def mul(self, a: Array, b: Array) -> Array:
        return self._mul(a, b)

    def sum(self, x: Array, *, axis: int) -> Array:
        # Field summation is XOR-reduce; one XLA reduce (log-depth, O(1) memory).
        # Identical to sumcheck._xor_reduce.
        return jax.lax.reduce(x, U64(0), jax.lax.bitwise_xor, (axis,))

    def domain_point(self, u: int, like: Array) -> Array:
        raise NotImplementedError(
            "flock's round message is the Karatsuba ∞-trick (r·G(1), G(∞)), not the "
            "degree+1-evals domain form — the round owns its message (see sumcheck.round_pair)."
        )

    def zeros_like(self, x: Array) -> Array:
        return jnp.zeros_like(x)
