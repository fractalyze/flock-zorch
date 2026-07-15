"""flock's product-sumcheck round, on zorch's compressed-domain round poly.

The multilinear rounds send `(s(1), s(∞))` — `s(∞)` the Karatsuba leading
coefficient sent instead of `s(2)` — observed as two framed scalars, then bind the
TOP bit at the squeezed challenge. The message is zorch's compressed product round
poly (`summand_evals` over `compressed_domain(1)`); the per-scalar framing is
flock-core's wire, so it observes each value on the SHA-256 transcript directly
(`observe_scalar` / `sample_scalar` — the framing distinction a Merlin/SHA-256
transcript needs, byte-matched to flock-core). Driven by `zorch.prove.fold_rounds`
inside one jit, a whole round loop is a single device program with FS inside.
"""
from __future__ import annotations

import functools

import frx

from zorch.prove import fold_rounds
from zorch.round import Round
from zorch.sumcheck.domain import compressed_domain, fold, summand_evals
from zorch.sumcheck.prover import ProductSummand

_PRODUCT2 = ProductSummand(2)._combine


class InfProductRound(Round):
    """One round over the stacked `[2, N]` ghash (weight, values) state: the
    `(s(1), s(∞))` compressed message observed as two framed scalars, the TOP bit
    bound at the squeezed challenge, returning `(s(1), s(∞), r)` so the caller keeps
    each round's challenge."""

    def __call__(self, folded, transcript):
        msg = summand_evals(folded, _PRODUCT2, compressed_domain(1, folded.dtype))
        transcript = transcript.observe_scalar(msg[0]).observe_scalar(msg[1])
        transcript, r = transcript.sample_scalar()
        return fold(folded, r), transcript, (msg[0], msg[1], r)


@functools.partial(frx.jit, static_argnums=(2,))
def prove_inf_product(stacked, transcript, rounds):
    """`rounds` ∞-product rounds as ONE device program:
    ([2, N], transcript) -> ([2, N/2^rounds], transcript, [(e1, einf, r), ...])."""
    return fold_rounds(InfProductRound(), stacked, transcript, rounds)
