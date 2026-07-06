"""flock's PCS in zorch's `PcsProver` seam shape (`zorch.pcs.protocol`):
`commit` binds one packed witness, `open` proves one ring-switched claim.

The threaded transcript is the flock `Challenger` — the byte-level host FS this
repo already threads through `ProveChain` (non-negotiable #3); the Array-level
`zorch.transcript.Transcript` does not apply, and a generic `observe` cannot
pick between flock's scalar/slice byte framings, so no conformance pin is made.
`values` = ring-switch `s_hat_v`: flock reduces f(point) to the 128
packing-lane partial evaluations, and the scalar claim is verifier-side
(`ring_switch::claim_check` against the consumer-supplied claim). The batched
dual-claim / Ligerito open assembly stays in `prover.py` (flock assembly).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from flock_zorch import field, pcs_commit, pcs_open
from flock_zorch.challenger import Challenger


@dataclass(frozen=True)
class FlockPcsProverData:
    """Retained commit output threaded into `open` (flock's `ProverData`)."""

    z_packed: Any
    codeword: np.ndarray
    tree: np.ndarray


@dataclass(frozen=True)
class FlockPcsProver:
    """`PcsProver`-shaped frontend over `pcs_commit.commit` / `pcs_open.open`.
    Commitment = the 32-byte Merkle root; proof = the open dict
    `{ring_switch, basefold}`."""

    m: int
    log_inv_rate: int
    log_batch_size: int
    mul: Callable = field.mul
    use_host_sha: bool = False

    @property
    def k_code(self) -> int:
        return (self.m - field.LOG_PACKING - self.log_batch_size) + self.log_inv_rate

    def commit(self, polys: Sequence[Any]) -> tuple[np.ndarray, FlockPcsProverData]:
        if len(polys) != 1:
            raise ValueError(
                f"flock's PCS commits exactly one packed witness, got {len(polys)}")
        z_packed = polys[0]
        expect = 1 << (self.m - field.LOG_PACKING)
        if len(z_packed) != expect:
            raise ValueError(
                f"packed witness has {len(z_packed)} positions, m={self.m} expects {expect}")
        root, codeword, tree = pcs_commit.commit(
            z_packed, self.m, self.log_inv_rate, self.log_batch_size,
            self.mul, self.use_host_sha)
        return root, FlockPcsProverData(z_packed=z_packed, codeword=codeword, tree=tree)

    def open(
        self,
        prover_data: FlockPcsProverData,
        points: Sequence[Any],
        transcript: Challenger,
    ) -> tuple[np.ndarray, dict, Challenger]:
        if len(points) != 1:
            raise ValueError(
                f"flock's PCS opens one ring-switched claim, got {len(points)}")
        proof = pcs_open.open(
            prover_data.z_packed, prover_data.codeword, prover_data.tree,
            points[0], self.k_code, self.log_inv_rate, self.log_batch_size,
            transcript, mul=self.mul, use_host_sha=self.use_host_sha)
        return proof["ring_switch"], proof, transcript
