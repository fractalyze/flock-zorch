# Copyright 2026 The flock-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The R1CS system's claims, witness and commitment data.

Both roles of a reduction read these, so neither imports the other to do it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from frx import Array

if TYPE_CHECKING:
    pass


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
class LigeritoCommitData:
    """What the Ligerito PCS retains between its commit and open halves.

    Not prover-only: the root is bound into the transcript by
    ``bind_statement`` and the opened columns and Merkle paths ride the proof.
    """

    root: Any
    pdata: Any


@dataclass(frozen=True, kw_only=True)
class ZerocheckClaim:
    """â, b̂ and ĉ evaluate to `a_eval`, `b_eval`, `c_eval` at the point the
    zerocheck drew — the skip scalar `z` together with the coordinate lists.

    Keyword-only: the three evals, and the two coordinate lists, are each
    interchangeable-looking, so a positional swap would type-check.
    """

    z: Any
    mlv_challenges: Any
    r_rest: Any
    a_eval: Any
    b_eval: Any
    c_eval: Any


@dataclass(frozen=True)
class ZerocheckProof:
    """flock's ZerocheckProof: the wire fields alone — the round-1 URM messages,
    the multilinear-round (G(1), G(∞)) pairs, and the final a/b/c evaluations.

    The evaluation point the proof is about lives on `ZerocheckClaim`, which
    `prove_packed` returns alongside this; holding a second copy here is what let
    the two disagree."""

    round1_ab: Any
    round1_c: Any
    multilinear_rounds: Any
    final_a_eval: Any
    final_b_eval: Any
    final_c_eval: Any


@dataclass(frozen=True)
class AbClaimPoint:
    """The â/b̂ evaluation point lincheck reduces — the zerocheck challenge split
    (flock's QuirkyPoint): `z_skip` the URM fold-point, `x_inner_rest` the inner
    multilinear challenges, `x_outer` the outer ones."""

    z_skip: Any
    x_inner_rest: Any
    x_outer: Any

    @classmethod
    def from_zerocheck(cls, zc: ZerocheckClaim, inner_rest: int) -> "AbClaimPoint":
        """The â/b̂ point derived from the zerocheck's claim: z_skip is the URM
        fold-point, and the multilinear challenges split into inner/outer at
        `inner_rest`."""
        return cls(
            z_skip=zc.z,
            x_inner_rest=zc.mlv_challenges[:inner_rest],
            x_outer=zc.mlv_challenges[inner_rest:],
        )


@dataclass(frozen=True)
class PackedDirectClaim:
    """A packed-direct PCS claim: a ẑ-evaluation `value` at `point` (its eq_ind is
    `build_eq(point)`), combined into the batched open alongside the ring-switched
    claims."""

    point: Any
    value: Any


@dataclass(frozen=True)
class LincheckClaim:
    """The post-sumcheck claim (flock prove_padded_inner steps 6-9): the fresh
    inner z_skip, the LSB-first inner-rest challenges, and the reduced value w."""

    r_inner_skip: Any
    r_inner_rest: Any
    w: Any


class LincheckProof(NamedTuple):
    """flock's lincheck proof: the product-sumcheck `rounds` and the `z_partial`
    message, plus the post-sumcheck `claim` (a `LincheckClaim`) the PCS open
    consumes. `z_vec` is not proof material — it is the initial partial fold
    (`partial_fold_packed_z`'s `[2^k_log]` output, flock's `z_vec_pre`) kept
    device-resident so the prover derives the ab claim's s_hat_v from it
    instead of re-reading the full packed witness (flock
    `prove_padded_capture_z_vec`). Access fields by attribute."""

    rounds: Any
    z_partial: Any
    claim: "LincheckClaim | None" = None
    z_vec: Any = None


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
    # The ab claim's precomputed s_hat_v (`[128]` ghash), derived by the
    # lincheck stage from its z_vec (`ring_switch.s_hat_v_from_z_vec` — flock
    # `ProveCore.s_hat_v_ab`): the open consumes it instead of re-reading the
    # full packed witness. None → the open falls back to the witness read.
    ab_s_hat_v: Any = None


@dataclass(frozen=True)
class BatchOpenProof:
    """Batched dual-claim PCS open (flock BatchOpeningProof): the per-claim
    ring-switch reductions plus the combined Ligerito low-degree open."""

    ring_switches: Any
    ligerito: Any = None
    ligerito_obj: Any = (
        None  # the zorch LigeritoProof (verify consumes this, not the wire dict)
    )


@dataclass(frozen=True)
class ProveFastResult:
    """flock's R1CS proof (`prover::prove`): the zerocheck and lincheck sub-proofs,
    the batched PCS open, and the final ab/c claim values."""

    zerocheck: Any
    lincheck: Any
    pcs_open: Any
    claim_ab_value: Any
    claim_c_value: Any
