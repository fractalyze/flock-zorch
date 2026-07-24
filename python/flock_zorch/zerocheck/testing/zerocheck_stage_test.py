"""Round-trip and soundness tests for the paired Flock zerocheck Stage."""
from __future__ import annotations

from dataclasses import replace

import frx
import frx.numpy as fnp
import numpy as np

frx.config.update("jax_enable_x64", True)

from flock_zorch import ghash  # noqa: E402
from flock_zorch.zerocheck.stage import ZerocheckStage, ZerocheckWitness  # noqa: E402
from flock_zorch.challenger import flock_transcript  # noqa: E402

DOMAIN = b"flock-zc-stage-test"
M = 13


def _transcript():
    return flock_transcript(DOMAIN)


def _witness():
    bits = (np.arange(1 << M, dtype=np.uint32) * 17 + 3) & 1
    return ZerocheckWitness(bits, bits, bits)


def _assert_claims_equal(left, right):
    for name in ("z", "mlv_challenges", "r_rest", "a_eval", "b_eval", "c_eval"):
        np.testing.assert_array_equal(
            ghash.to_lanes(getattr(left, name)),
            ghash.to_lanes(getattr(right, name)),
            err_msg=name,
        )


def test_zerocheck_stage_roundtrip():
    stage = ZerocheckStage(M)
    proved = stage.prove(_witness(), _transcript())
    verified = stage.verify(None, proved.proof, _transcript())
    assert bool(np.asarray(verified.ok))
    _assert_claims_equal(proved.output, verified.output)


def test_zerocheck_stage_rejects_tampered_final_claim():
    stage = ZerocheckStage(M)
    proof = stage.prove(_witness(), _transcript()).proof
    one = ghash.to_ghash(fnp.asarray([1, 0], fnp.uint64))
    bad = replace(proof, final_a_eval=proof.final_a_eval + one)
    verified = stage.verify(None, bad, _transcript())
    assert not bool(np.asarray(verified.ok))
