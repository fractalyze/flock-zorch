"""flock-zorch reuses zorch's device sumcheck driver — over flock's GHASH field.

This is a reuse/architecture test: it proves flock's sumcheck shape (a
product-of-multilinears round, and the ∞-trick round that sends only
(s(1), leading coeff)) drives zorch's `sumcheck.prove` `lax.scan` and its
verifier, end to end, over `binary_field_ghash` — flock's exact GF(2¹²⁸)
basis, jax-native since jaxlib dev2026-07-06 (xla#169 + jax#82).

Unlike the retired koalabear stand-in, GHASH has no transcript-boundary
soundness gap: raw-byte challenge sampling is canonical for a binary field
(every byte pattern is a valid element), so the `Sha256FieldTranscript` test
asserts full soundness in flock's real configuration (∞-domain + per-element
scalar framing + SHA-256 byte transcript). Byte-identity against flock-core
goldens is NOT asserted here — that lands with the zerocheck/lincheck port.

Run on the venv (not a bazel gate — like field_test.py); backend from JAX_PLATFORMS:
    export PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)"
    .venv/bin/python python/flock_zorch/testing/zorch_sumcheck_reuse_test.py
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import zk_dtypes  # noqa: E402

from zorch.sha256_field_transcript import Sha256FieldTranscript  # noqa: E402
from zorch.sumcheck.prover import INF, SumcheckRound, prove  # noqa: E402
from zorch.sumcheck.verifier import (  # noqa: E402
    InfDomainSumcheckRound,
    SumcheckRound as VerifySumcheckRound,
)
from zorch.sumcheck.testing import eval_mle_oracle, product  # noqa: E402
from zorch.testkit.transcript import cheap_transcript  # noqa: E402

GHASH = zk_dtypes.binary_field_ghash


def rand_ghash(seed: int, shape: tuple[int, ...]) -> jax.Array:
    """Full-width random GHASH elements. Char-2 fields have no modulus, so any
    16-byte pattern is a valid canonical element — draw uint64 lane pairs and
    reinterpret (zorch's `rand_field` draws canonical ints < 2^30, which for a
    128-bit binary field would leave 98 bits always-zero)."""
    raw = np.random.default_rng(seed).bytes(int(np.prod(shape)) * 16)
    return jnp.asarray(np.frombuffer(raw, dtype=np.uint8).view(GHASH).reshape(shape))


def _verify(vround, claim, proof, n):
    """Thread the verifier round over the proof: returns (final_claim, challenges).

    Each `vround` call checks its round identity (`ok`), reduces the claim, and
    advances the same transcript the prover used, so a Fiat-Shamir divergence or a
    bad round message fails here."""
    transcript, challenges = cheap_transcript(GHASH), []
    for i in range(n):
        claim, transcript, r, ok = vround(claim, proof[i], transcript)
        assert bool(ok), f"round {i} identity failed"
        challenges.append(r)
    return claim, jnp.stack(challenges)


def test_natural_domain_reuse():
    """Default (natural [0..degree]) product sumcheck through zorch's driver +
    verifier, over GHASH."""
    n = 8
    factors = [rand_ghash(22, (1 << n,)), rand_ghash(23, (1 << n,))]
    _, _, msgs = prove(SumcheckRound(degree=2), factors, cheap_transcript(GHASH))
    assert msgs.round_poly.shape == (n, 3)  # [s(0), s(1), s(2)] per round

    final_claim, challenges = _verify(VerifySumcheckRound(degree=2), jnp.sum(product(factors)), msgs.round_poly, n)
    want = product([eval_mle_oracle(f, challenges) for f in factors])
    assert bool(final_claim == want)


def test_inf_domain_reuse():
    """Round-owned ∞-trick domain (1, INF): the round sends only (s(1), s(∞)) and
    the verifier recovers s(0) from the running claim. This is flock's round_pair
    message shape, over flock's field."""
    n = 8
    factors = [rand_ghash(101, (1 << n,)), rand_ghash(202, (1 << n,))]
    final_state, _, msgs = prove(
        SumcheckRound(degree=2, domain=(1, INF)), factors, cheap_transcript(GHASH)
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
    factors = [rand_ghash(7, (1 << n,)), rand_ghash(9, (1 << n,))]
    _, _, natural = prove(SumcheckRound(degree=2), factors, cheap_transcript(GHASH))
    _, _, inf = prove(SumcheckRound(degree=2, domain=(1, INF)), factors, cheap_transcript(GHASH))
    assert natural.round_poly.shape == (n, 3)
    assert inf.round_poly.shape == (n, 2)


def test_scalar_framing_soundness():
    """flock's actual round configuration — ∞-trick domain, per-element scalar
    framing (`Challenger.observe_f128`, no length prefix), SHA-256 byte
    transcript — asserted for FULL soundness. Raw-byte challenge squeezing is
    canonical over a binary field, so unlike the retired koalabear stand-in the
    final reduced claim must equal the multilinear oracle evaluation, not just
    stay in Fiat-Shamir lockstep."""
    n = 6
    factors = [rand_ghash(11, (1 << n,)), rand_ghash(12, (1 << n,))]
    _, _, msgs = prove(
        SumcheckRound(degree=2, domain=(1, INF), scalar_framing=True),
        factors,
        Sha256FieldTranscript.new(b"flock", GHASH),
    )
    assert msgs.round_poly.shape == (n, 2)

    transcript, challenges = Sha256FieldTranscript.new(b"flock", GHASH), []
    claim = jnp.sum(product(factors))
    for i in range(n):
        claim, transcript, r, ok = InfDomainSumcheckRound(
            degree=2, scalar_framing=True
        )(claim, msgs.round_poly[i], transcript)
        assert bool(ok), f"round {i} identity failed"
        challenges.append(r)
    assert bool(jnp.array_equal(msgs.challenge, jnp.stack(challenges)))
    want = product([eval_mle_oracle(f, challenges) for f in factors])
    assert bool(claim == want)


if __name__ == "__main__":
    test_natural_domain_reuse()
    test_inf_domain_reuse()
    test_default_domain_unchanged()
    test_scalar_framing_soundness()
    print(f"zorch sumcheck reuse over GHASH: PASS on {jax.default_backend()}")
