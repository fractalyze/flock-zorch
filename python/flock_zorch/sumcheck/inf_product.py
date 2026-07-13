"""flock's product-sumcheck round over the TOP-bit split, with the ∞-trick wire.

flock's multilinear rounds send (s(1), s(∞)) — s(∞) is the Karatsuba leading
coefficient, sent instead of s(2) — as two scalar-framed observes, then bind the
TOP bit at the squeezed challenge. The round math is zorch's MSB split/fold
(`fold` = P0 + r·(P1−P0), which over char 2 is flock's `sumcheck_bind_top`);
only the message wire is flock's. Driven by `zorch.prove.fold_rounds` inside one
jit, a whole round loop is a single device program with Fiat-Shamir inside.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.sumcheck.domain import fold, split_halves


class InfProductRound(Round):
    """One round over the stacked [2, N] ghash (weight, values) state:
    message (s(1), s(∞)) = (Σ w_hi·v_hi, Σ (w_hi+w_lo)·(v_hi+v_lo)), observed
    scalar-framed in that order (flock `sumcheck_round_eval`), then fold at the
    squeezed challenge."""

    def __call__(self, folded, transcript):
        lo, hi = split_halves(folded)
        e1 = jnp.sum(hi[0] * hi[1])
        einf = jnp.sum((hi[0] + lo[0]) * (hi[1] + lo[1]))
        transcript = transcript.observe_scalar(e1).observe_scalar(einf)
        transcript, r = transcript.sample_scalar()
        return fold(folded, r), transcript, (e1, einf, r)


@functools.partial(jax.jit, static_argnums=(2,))
def prove_inf_product(stacked, transcript, rounds):
    """`rounds` ∞-product rounds as ONE device program:
    ([2, N], transcript) -> ([2, N/2^rounds], transcript, [(e1, einf, r), ...])."""
    return fold_rounds(InfProductRound(), stacked, transcript, rounds)
