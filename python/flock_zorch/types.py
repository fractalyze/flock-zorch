# Copyright 2026 The flock-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The R1CS system's claims, witness and commitment data.

Separate from `prover.py` so a verifier reads a claim without importing the
prover that produced it. The two roles of a claim reduction are separately
deployable (`zorch.stage`), which a shared type module is what makes possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from frx import Array

if TYPE_CHECKING:
    from flock_zorch.pcs.ligerito import LigeritoConfig


@dataclass(frozen=True, kw_only=True)
class R1csClaim:
    """The committed witness ẑ satisfies the R1CS instance: A·ẑ ∘ B·ẑ = C·ẑ.

    The root statement a flock proof discharges. It carries the instance digest,
    which is what the transcript binds and what pins *which* R1CS is being
    claimed; the matrix shapes are fixed by the circuit and configure the roles.
    """

    statement_digest: bytes


@dataclass(frozen=True, kw_only=True)
class R1csWitness:
    """The assignment ẑ that satisfies the R1CS claim, in the two forms the
    reductions consume: packed bits for the zerocheck and PCS, bytes plus the
    dense A/B blocks for the lincheck."""

    z_packed: Array  # witness ẑ, device-resident across the reductions
    z_lincheck: bytes
    a0: Array  # lincheck A matrix (dense)
    b0: Array  # lincheck B matrix (dense)


@dataclass(frozen=True, kw_only=True)
class BatchOpeningClaim:
    """ẑ opens to the ab and c claim values at the two batched points.

    What the lincheck leaves for the PCS: two evaluation claims on the committed
    witness, which the batched Ligerito open discharges.
    """

    ab_point: Array
    c_point: Array
    ab_value: Any
    c_value: Any


@dataclass(frozen=True, kw_only=True)
class LigeritoCommitData:
    """What the Ligerito PCS retains between its commit and open halves.

    Not prover-only: the root is bound into the transcript by
    ``bind_statement`` and the opened columns and Merkle paths ride the proof.
    """

    root: Any
    pdata: Any
