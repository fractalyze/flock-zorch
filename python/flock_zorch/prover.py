"""flock's fused R1CS prover (`prover::prove` / `prove_fast_core`), authored in
frx — byte-identical to flock-core. Two claim reductions bracketed by the
Ligerito PCS, threading ONE
shared SHA-256 challenger with device-resident state (no per-phase host
re-transfer): commit+bind → zerocheck → lincheck → batched PCS open (see
`prove_fast`).

a = A·z, b = B·z are kept device-resident across the phases (no per-phase witness
re-transfer). Gated by `testing/e2e_ligerito_oracle_test.py` against flock
`prover::prove_fast_ligerito`.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from zorch.stage import ProveResult, ProverStage, TrivialClaim

from flock_zorch import ghash
from flock_zorch.challenger import Challenger  # noqa: F401  (re-exported for callers)
from flock_zorch.lincheck.prover import LincheckProver
from flock_zorch.pcs import ligerito as zorch_ligerito
from flock_zorch.pcs import ring_switch
from flock_zorch.sumcheck import build_eq
from flock_zorch.types import (
    BatchOpeningClaim,
    BatchOpenProof,
    LigeritoCommitData,
    ProveFastResult,
    R1csClaim,
    R1csWitness,
)
from flock_zorch.zerocheck.prover import ZerocheckProver


@frx.jit
def _unpack_bits(z_packed):
    """Packed F128 witness [2^(m-7),2] -> device bit witness [2^m] uint8 (LSB-first
    within each 128-bit element), on device so a=b=c=z stays device-resident. The
    frx analogue of flock's `pcs::pack::unpack_witness`."""
    bitidx = fnp.arange(64, dtype=fnp.uint64)
    lo = ((z_packed[:, 0:1] >> bitidx) & fnp.uint64(1)).astype(fnp.uint8)
    hi = ((z_packed[:, 1:2] >> bitidx) & fnp.uint64(1)).astype(fnp.uint8)
    return fnp.concatenate([lo, hi], axis=1).reshape(-1)


def _as_bytes(x) -> bytes:
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    return np.asarray(x, np.uint8).tobytes()


def bind_statement(ch, statement_digest, root) -> None:
    """Bind the Fiat-Shamir transcript to the statement (flock `proof::bind_statement`):
    observe `flock-r1cs-v0` + the R1CS instance digest + the commitment root. Call
    once after commit, before any sub-protocol challenge."""
    ch.observe_label(b"flock-r1cs-v0")
    ch.observe_bytes(_as_bytes(statement_digest))
    ch.observe_bytes(_as_bytes(root))


def _combine_claims(
    rs_eq_inds, gammas, sumcheck_claims, packed_direct=(), gammas_pd=()
):
    """γ-combine the batched ring-switch claims (+ optional packed-direct claims) into
    the single (b_combined, target) the Ligerito open runs against. The
    ring-switch γ's are already baked into each rs_eq_ind by prove_batched, so b is
    their XOR-sum; target = Σ γ_i·sumcheck_claim_i. Packed-direct claims add
    γ_pd_j·eq(point_j) to b and γ_pd_j·value_j to target. NB: all observe/sample stay
    at the call sites — this is pure arithmetic, so it cannot perturb the transcript."""
    b_combined = rs_eq_inds[0]  # native ghash [2^L]
    for r in rs_eq_inds[1:]:
        b_combined = b_combined + r  # γ_rs already baked in
    target = ghash.to_ghash(fnp.zeros(2, fnp.uint64))  # ghash scalar zero
    for g, sc in zip(gammas, sumcheck_claims):  # both native ghash
        target = target + g * sc
    for pd, g in zip(packed_direct, gammas_pd):  # g native ghash
        eq_pd = build_eq(pd.point)  # length L = 2^(m-7)
        b_combined = b_combined + g * eq_pd
        target = target + g * pd.value  # pd.value native ghash
    return b_combined, target  # native ghash: [2^L], scalar


def open_batch_ligerito(config, z_packed, pdata, x_outers, ch) -> BatchOpenProof:
    """Batched dual-claim PCS open with the LIGERITO backend — the headline path.
    The no-packed-direct case of `open_batch_mixed_ligerito`: N ring-switched
    claims (x_outers, e.g. ab+c), zero direct ẑ-evaluation claims. `pdata` is the
    ligerito commit from `zorch_ligerito.commit_flock_ligerito`. Returns
    {ring_switches, ligerito: LigeritoProof}."""
    return open_batch_mixed_ligerito(config, z_packed, pdata, x_outers, (), ch)


def open_batch_mixed_ligerito(
    config, z_packed, pdata, x_outers, packed_direct, ch
) -> BatchOpenProof:
    """Mixed batched open (flock `open_batch_mixed_ligerito_with_precomputed_s_hat_v`)
    — the HASH-CHAIN open, and the general Ligerito open. Combines N ring-switched
    claims (x_outers, e.g. ab+c) with M packed-direct claims (the chain claim: a
    direct ẑ-evaluation at a point, eq_ind = build_eq(point) == build_eq_sparse(point)).
    The combine is Σ_i γ_i·rs_eq_ind_i → b_combined (target Σ_i γ_i·sumcheck_claim_i),
    then b_combined gains Σ_j γ_pd_j·eq_ind_j and the target Σ_j γ_pd_j·value_j; the
    recursive Ligerito prover runs against (b_combined, target). γ order: the
    ring-switch γ's first (sampled inside prove_batched), then γ_pd after observing
    each packed-direct value. M=0 recovers the plain Ligerito open
    (open_batch_ligerito).
    `pdata` is the ligerito commit reused from the commit phase (no L0 re-encode)."""
    ch.observe_label(b"flock-pcs-open-batch-v0")
    s_hat_vs, rs_eq_inds, sumcheck_claims, gammas = ring_switch.prove_batched(
        z_packed, x_outers, ch
    )
    # Packed-direct: observe each claim's value, THEN sample the γ_pd (flock order).
    for pd in packed_direct:
        ch.observe_label(b"flock-pcs-packed-direct-v0")
        ch.observe_f128(pd.value)  # native ghash scalar
    gammas_pd = [ch.sample_f128() for _ in packed_direct]

    b_combined, target = _combine_claims(
        rs_eq_inds,
        gammas,
        sumcheck_claims,
        packed_direct=packed_direct,
        gammas_pd=gammas_pd,
    )
    # The Ligerito recursion runs in zorch (`zorch.pcs.ligerito`) via the flock
    # FS seam, reusing the commit-phase `pdata` directly. The ghash algebra rides
    # the dtype, so `mul` is not threaded.
    lig, lig_obj = zorch_ligerito.prove_flock_ligerito(
        config, pdata, b_combined, target, ch, return_proof=True
    )
    return BatchOpenProof(ring_switches=s_hat_vs, ligerito=lig, ligerito_obj=lig_obj)


class FlockLigeritoPcs:
    """The Ligerito commitment scheme: commit ẑ, open it at the batched points.

    Two halves of one role, held apart by Fiat-Shamir — the commitment must bind
    the transcript before the zerocheck draws a challenge, and the open needs
    the points the lincheck produces. ``LigeritoCommitData`` names what crosses
    between them.

    Discharges the batched opening claim into the trivial claim: the open is
    terminal, so a flock proof is a complete argument rather than one link in a
    chain.
    """

    def __init__(self, cfg):
        self._cfg = cfg

    def commit(self, witness: R1csWitness) -> LigeritoCommitData:
        root, pdata = zorch_ligerito.commit_flock_ligerito(self._cfg, witness.z_packed)
        return LigeritoCommitData(root=root, pdata=pdata)

    def prove(
        self,
        claim: BatchOpeningClaim,
        witness: tuple[R1csWitness, LigeritoCommitData],
        transcript,
    ) -> ProveResult[TrivialClaim, BatchOpenProof]:
        r1cs_witness, commit_data = witness
        proof = open_batch_ligerito(
            self._cfg,
            r1cs_witness.z_packed,
            commit_data.pdata,
            [claim.ab_point, claim.c_point],
            transcript,
        )
        return ProveResult(TrivialClaim(), proof, transcript)


class FlockProver(ProverStage[R1csClaim, R1csWitness, TrivialClaim, ProveFastResult]):
    """flock's R1CS prover: the Ligerito commit, then two reductions, then the
    batched open.

    One definition of the wiring, so `prove_fast` and the oracle gates cannot
    drift on it. `LigeritoCommitData` is held here between the PCS halves
    because it belongs to neither claim.
    """

    def __init__(self, cfg, m, k_log, k_skip, circuit=None):
        self.pcs = FlockLigeritoPcs(cfg)
        self.zerocheck = ZerocheckProver(m)
        self.lincheck = LincheckProver(m, k_log, k_skip, circuit)

    def prove(
        self, claim: R1csClaim, witness: R1csWitness, transcript
    ) -> ProveResult[TrivialClaim, ProveFastResult]:
        commit_data = self.pcs.commit(witness)
        # A shared function both roles call between the PCS halves: it only
        # absorbs, so it owns no proof section of its own.
        bind_statement(transcript, claim.statement_digest, commit_data.root)
        zc = self.zerocheck.prove(claim, witness, transcript)
        lc = self.lincheck.prove(zc.reduced_claim, witness, zc.transcript)
        opening = self.pcs.prove(
            lc.reduced_claim, (witness, commit_data), lc.transcript
        )
        return ProveResult(
            TrivialClaim(),
            ProveFastResult(
                zerocheck=zc.reduction_proof,
                lincheck=lc.reduction_proof,
                pcs_open=opening.reduction_proof,
                claim_ab_value=lc.reduced_claim.ab_value,
                claim_c_value=lc.reduced_claim.c_value,
            ),
            opening.transcript,
        )


def prove_fast(
    z_packed: Array,
    m: int,
    k_log: int,
    k_skip: int,
    a0: Array,
    b0: Array,
    z_lincheck: bytes,
    statement_digest: bytes,
    cfg,
    circuit=None,
    domain: bytes = b"flock-test-v0",
) -> ProveFastResult:
    """Fused single-call R1CS prover on the Ligerito PCS, byte-identical to flock
    `prover::prove_fast_ligerito`. Drives `FlockProver` — the Ligerito commit and
    statement bind, then the zerocheck and lincheck reductions, then the batched
    dual-claim Ligerito open — threading one shared challenger (no per-phase host
    re-transfer). `cfg` is the flock Ligerito config; `circuit` a
    `LincheckCircuit` for real hash R1CS (None uses the dense a0/b0 path — the
    identity gate). a = A·z, b = B·z; for the identity R1CS a = b = c = z."""
    prover = FlockProver(cfg, m, k_log, k_skip, circuit)
    return prover.prove(
        R1csClaim(statement_digest=statement_digest),
        R1csWitness(z_packed=z_packed, z_lincheck=z_lincheck, a0=a0, b0=b0),
        Challenger(domain),
    ).reduction_proof
