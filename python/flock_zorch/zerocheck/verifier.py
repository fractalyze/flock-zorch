"""Paired verifier for flock's univariate-skip zerocheck."""
from __future__ import annotations

import numpy as np
import frx.numpy as fnp

from flock_zorch import ghash
from flock_zorch.challenger import FlockTranscript, flock_transcript
from flock_zorch.zerocheck._fold import (
    _batch_inv,
    _interpolate_at_z_combined,
    _interpolate_at_z_on_lambda_g,
)
from flock_zorch.zerocheck.prover import (
    K_SKIP,
    LABEL,
    N_INNER,
    ZerocheckClaim,
    ZerocheckProof,
    _MEDIUM_G,
    _SMALL_G,
)

_ONE_G = ghash.to_ghash(fnp.asarray([1, 0], fnp.uint64))


def _as_ghash(value):
    value = fnp.asarray(value)
    if np.dtype(value.dtype) == np.dtype(fnp.binary_field_ghash):
        return value
    return ghash.to_ghash(value)


def verify(
    proof: ZerocheckProof,
    m: int,
    *,
    domain: bytes | None = None,
    transcript: FlockTranscript | None = None,
) -> tuple[ZerocheckClaim, FlockTranscript, object]:
    """Replay the Flock verifier, returning its semantic claim and verdict."""
    k_skip = K_SKIP
    if m < k_skip + N_INNER:
        raise ValueError(f"m must be >= {k_skip + N_INNER}, got {m}")
    ell = 1 << k_skip
    n_mlv = m - k_skip
    if fnp.shape(proof.round1_ab)[0] != ell:
        raise ValueError(f"round1_ab needs {ell} elements")
    if fnp.shape(proof.round1_c)[0] != ell:
        raise ValueError(f"round1_c needs {ell} elements")
    if len(proof.multilinear_rounds) != n_mlv:
        raise ValueError(f"need {n_mlv} multilinear rounds")
    if transcript is None:
        if domain is None:
            raise ValueError("domain is required when no transcript is supplied")
        transcript = flock_transcript(domain)

    transcript = transcript.observe_label(LABEL)
    transcript, r_skip = transcript.sample_f128(k_skip)
    transcript, r_outer = transcript.sample_f128(m - k_skip - N_INNER)
    r = fnp.concatenate([r_skip, _SMALL_G, _MEDIUM_G, r_outer])

    round1_ab = _as_ghash(proof.round1_ab)
    round1_c = _as_ghash(proof.round1_c)
    transcript = transcript.observe_f128(round1_ab)
    transcript = transcript.observe_f128(round1_c)
    transcript, z = transcript.sample_f128()

    computed_c = _interpolate_at_z_on_lambda_g(proof.round1_c, k_skip, z)
    final_c = _as_ghash(proof.final_c_eval)
    ok = computed_c == final_c

    combined_at_z = _interpolate_at_z_combined(
        ghash.from_ghash(round1_ab + round1_c), k_skip, z
    )
    running = combined_at_z + computed_c
    rhos = []
    for i, (msg_1, msg_inf) in enumerate(proof.multilinear_rounds):
        r_eq = r[k_skip + i]
        one_plus_r_eq = _ONE_G + r_eq
        g1 = _as_ghash(msg_1)
        g_inf = _as_ghash(msg_inf)
        inv = _batch_inv(one_plus_r_eq.reshape(1))[0]
        g0 = (running + r_eq * g1) * inv

        transcript = transcript.observe_f128(g1)
        transcript = transcript.observe_f128(g_inf)
        transcript, rho = transcript.sample_f128()
        rhos.append(rho)
        one_plus_rho = _ONE_G + rho
        running = (
            g0 * one_plus_rho
            + g1 * rho
            + g_inf * rho * one_plus_rho
        )

    final_a = _as_ghash(proof.final_a_eval)
    final_b = _as_ghash(proof.final_b_eval)
    ok = ok & (running == final_a * final_b)
    transcript = transcript.observe_f128(final_a)
    transcript = transcript.observe_f128(final_b)

    claim = ZerocheckClaim(
        z=z,
        mlv_challenges=fnp.stack(rhos),
        r_rest=r[k_skip:],
        a_eval=final_a,
        b_eval=final_b,
        c_eval=final_c,
    )
    return claim, transcript, ok
