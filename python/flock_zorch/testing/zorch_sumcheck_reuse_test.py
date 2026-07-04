"""flock-zorch reuses zorch's device sumcheck driver — validated over koalabear.

This is a reuse/architecture test, NOT a flock byte-match oracle: it proves flock's
sumcheck shape (a product-of-multilinears round, and the ∞-trick round that sends
only (s(1), leading coeff)) drives zorch's `sumcheck.prove` `lax.scan` and its
verifier, end to end.

koalabear stands in for flock's GHASH GF(2^128): binary-field dtypes are not jax-
lowerable (numpy-only), so the device driver — a jax program — cannot yet carry
flock's field. koalabear is a first-class jax field, so it exercises the whole
reuse ARCHITECTURE (round seam, device scan, transcript, verifier) today; the swap
to GHASH + additive NTT waits on binary-field GPU readiness (milestone P4/P5). No
byte-identity is asserted here (there is no koalabear flock golden).

Run on the venv (not a bazel gate — like field_test.py); backend from JAX_PLATFORMS:
    export PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)"
    .venv/bin/python python/flock_zorch/testing/zorch_sumcheck_reuse_test.py
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import zk_dtypes  # noqa: E402

from zorch.sumcheck.prover import INF, SumcheckRound, prove  # noqa: E402
from zorch.sumcheck.verifier import (  # noqa: E402
    InfDomainSumcheckRound,
    SumcheckRound as VerifySumcheckRound,
)
from zorch.sumcheck.testing import eval_mle_oracle, product  # noqa: E402
from zorch.testkit.random_field import rand_field  # noqa: E402
from zorch.testkit.transcript import cheap_transcript  # noqa: E402

KB = zk_dtypes.koalabear_mont


def _verify(vround, claim, proof, n):
    """Thread the verifier round over the proof: returns (final_claim, challenges).

    Each `vround` call checks its round identity (`ok`), reduces the claim, and
    advances the same transcript the prover used, so a Fiat-Shamir divergence or a
    bad round message fails here."""
    transcript, challenges = cheap_transcript(KB), []
    for i in range(n):
        claim, transcript, r, ok = vround(claim, proof[i], transcript)
        assert bool(ok), f"round {i} identity failed"
        challenges.append(r)
    return claim, jnp.stack(challenges)


def test_natural_domain_reuse():
    """Default (natural [0..degree]) product sumcheck through zorch's driver +
    verifier, over koalabear."""
    n = 8
    factors = [rand_field(22, (1 << n,), KB), rand_field(23, (1 << n,), KB)]
    _, _, msgs = prove(SumcheckRound(degree=2), factors, cheap_transcript(KB))
    assert msgs.round_poly.shape == (n, 3)  # [s(0), s(1), s(2)] per round

    final_claim, challenges = _verify(VerifySumcheckRound(degree=2), jnp.sum(product(factors)), msgs.round_poly, n)
    want = product([eval_mle_oracle(f, challenges) for f in factors])
    assert bool(final_claim == want)


def test_inf_domain_reuse():
    """Round-owned ∞-trick domain (1, INF): the round sends only (s(1), s(∞)) and
    the verifier recovers s(0) from the running claim. This is flock's round_pair
    message shape, field-agnostic (koalabear here)."""
    n = 8
    factors = [rand_field(101, (1 << n,), KB), rand_field(202, (1 << n,), KB)]
    final_state, _, msgs = prove(
        SumcheckRound(degree=2, domain=(1, INF)), factors, cheap_transcript(KB)
    )
    assert msgs.round_poly.shape == (n, 2)  # (s(1), s(inf)) — 2 elements, not 3

    final_claim, challenges = _verify(InfDomainSumcheckRound(degree=2), jnp.sum(product(factors)), msgs.round_poly, n)
    # FS challenges the prover derived must match the verifier's.
    assert bool(jnp.all(msgs.challenge == challenges))
    want = product([eval_mle_oracle(f, challenges) for f in factors])
    assert bool(final_claim == want)
    assert bool(product([s[0] for s in final_state]) == want)


def test_default_domain_unchanged():
    """domain=None keeps the natural 3-value wire form (the byte-identical default
    path for existing zorch callers)."""
    n = 4
    factors = [rand_field(7, (1 << n,), KB), rand_field(9, (1 << n,), KB)]
    _, _, natural = prove(SumcheckRound(degree=2), factors, cheap_transcript(KB))
    _, _, inf = prove(SumcheckRound(degree=2, domain=(1, INF)), factors, cheap_transcript(KB))
    assert natural.round_poly.shape == (n, 3)
    assert inf.round_poly.shape == (n, 2)


if __name__ == "__main__":
    test_natural_domain_reuse()
    test_inf_domain_reuse()
    test_default_domain_unchanged()
    print(f"zorch sumcheck reuse over koalabear: PASS on {jax.default_backend()}")
