"""Flock zerocheck as an ordinary paired zorch Stage."""
from __future__ import annotations

from dataclasses import dataclass

from zorch.stage import ProveResult, Stage, VerifyResult
from zorch.transcript import Transcript

from flock_zorch import ghash
from flock_zorch.challenger import FlockTranscript
from flock_zorch.zerocheck.prover import (
    ZerocheckClaim,
    ZerocheckProof,
    prove_packed,
)
from flock_zorch.zerocheck.verifier import verify as verify_proof


@dataclass(frozen=True)
class ZerocheckWitness:
    a_bits: object
    b_bits: object
    c_bits: object


class ZerocheckStage(
    Stage[
        ZerocheckWitness,
        ZerocheckClaim,
        None,
        ZerocheckClaim,
        ZerocheckProof,
    ]
):
    """The complete Flock URM-plus-multilinear zerocheck component."""

    name = "flock_zerocheck"

    def __init__(self, m: int) -> None:
        self.m = m

    @staticmethod
    def _transcript(transcript: Transcript) -> FlockTranscript:
        if not isinstance(transcript, FlockTranscript):
            raise TypeError("Flock zerocheck needs FlockTranscript")
        return transcript

    def prove(
        self, inputs: ZerocheckWitness, transcript: Transcript
    ) -> ProveResult[ZerocheckClaim, ZerocheckProof]:
        transcript = self._transcript(transcript)
        proof, transcript = prove_packed(
            inputs.a_bits,
            inputs.b_bits,
            inputs.c_bits,
            self.m,
            transcript=transcript,
        )
        claim = ZerocheckClaim(
            z=proof.z,
            mlv_challenges=proof.mlv_challenges,
            r_rest=proof.r_rest,
            a_eval=proof.final_a_eval,
            b_eval=proof.final_b_eval,
            c_eval=ghash.to_ghash(proof.final_c_eval),
        )
        return ProveResult(claim, proof, transcript)

    def verify(
        self,
        inputs: None,
        proof: ZerocheckProof,
        transcript: Transcript,
    ) -> VerifyResult[ZerocheckClaim]:
        del inputs
        transcript = self._transcript(transcript)
        claim, transcript, ok = verify_proof(
            proof, self.m, transcript=transcript
        )
        return VerifyResult(claim, transcript, ok)
