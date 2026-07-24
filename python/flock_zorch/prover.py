"""flock's fused R1CS prover (`prover::prove` / `prove_fast_core`), authored in
frx — byte-identical to flock-core. The protocol's heterogeneous phases are
orchestrated explicitly while threading one shared SHA-256 challenger:
commit+bind → zerocheck → lincheck → batched PCS open (see `prove_fast`).

a = A·z, b = B·z are kept device-resident across the phases (no per-phase witness
re-transfer). Gated by `testing/e2e_ligerito_oracle_test.py` against flock
`prover::prove_fast_ligerito`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import frx
import frx.numpy as fnp
from frx import Array

from flock_zorch import ghash, zerocheck, lincheck
from flock_zorch.pcs import ring_switch, ligerito as zorch_ligerito
from flock_zorch.sumcheck import build_eq
from flock_zorch.challenger import FlockTranscript, flock_transcript


@dataclass(frozen=True)
class BatchOpenProof:
    """Batched dual-claim PCS open (flock BatchOpeningProof): the per-claim
    ring-switch reductions plus the combined Ligerito low-degree open."""

    ring_switches: Any
    ligerito: Any = None


@dataclass(frozen=True)
class ProveFastResult:
    """flock's R1CS proof (`prover::prove`): the zerocheck and lincheck sub-proofs,
    the batched PCS open, and the final ab/c claim values."""

    zerocheck: Any
    lincheck: Any
    pcs_open: Any
    claim_ab_value: Any
    claim_c_value: Any


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


def bind_statement(
    transcript: FlockTranscript, statement_digest, root
) -> FlockTranscript:
    """Bind the Fiat-Shamir transcript to the statement (flock `proof::bind_statement`):
    observe `flock-r1cs-v0` + the R1CS instance digest + the commitment root. Call
    once after commit, before any sub-protocol challenge."""
    transcript = transcript.observe_label(b"flock-r1cs-v0")
    transcript = transcript.observe_bytes(_as_bytes(statement_digest))
    return transcript.observe_bytes(_as_bytes(root))


def _combine_claims(rs_eq_inds, gammas, sumcheck_claims, packed_direct=(), gammas_pd=()):
    """γ-combine the batched ring-switch claims (+ optional packed-direct claims) into
    the single (b_combined, target) the Ligerito open runs against. The
    ring-switch γ's are already baked into each rs_eq_ind by prove_batched, so b is
    their XOR-sum; target = Σ γ_i·sumcheck_claim_i. Packed-direct claims add
    γ_pd_j·eq(point_j) to b and γ_pd_j·value_j to target. NB: all observe/sample stay
    at the call sites — this is pure arithmetic, so it cannot perturb the transcript."""
    b_combined = rs_eq_inds[0]                                     # native ghash [2^L]
    for r in rs_eq_inds[1:]:
        b_combined = b_combined + r                                # γ_rs already baked in
    target = ghash.to_ghash(fnp.zeros(2, fnp.uint64))              # ghash scalar zero
    for g, sc in zip(gammas, sumcheck_claims):                     # both native ghash
        target = target + g * sc
    for pd, g in zip(packed_direct, gammas_pd):                    # g native ghash
        eq_pd = build_eq(ghash.to_ghash(fnp.asarray(pd.point)))   # length L = 2^(m-7)
        b_combined = b_combined + g * eq_pd
        target = target + g * pd.value                             # pd.value native ghash
    return b_combined, target  # native ghash: [2^L], scalar


def open_batch_ligerito(config, z_packed, pdata, x_outers, transcript):
    """Batched dual-claim PCS open with the LIGERITO backend — the headline path.
    The no-packed-direct case of `open_batch_mixed_ligerito`: N ring-switched
    claims (x_outers, e.g. ab+c), zero direct ẑ-evaluation claims. `pdata` is the
    ligerito commit from `zorch_ligerito.commit_flock_ligerito`. Returns the
    batch-opening proof and advanced transcript."""
    return open_batch_mixed_ligerito(
        config, z_packed, pdata, x_outers, (), transcript
    )


def open_batch_mixed_ligerito(
    config, z_packed, pdata, x_outers, packed_direct, transcript
):
    """Mixed batched open (flock `open_batch_mixed_ligerito_with_precomputed_s_hat_v`)
    — the HASH-CHAIN open, and the general Ligerito open. Combines N ring-switched
    claims (x_outers, e.g. ab+c) with M packed-direct claims (the chain claim: a
    direct ẑ-evaluation at a point, eq_ind = build_eq(point) == build_eq_sparse(point)).
    The combine is Σ_i γ_i·rs_eq_ind_i → b_combined (target Σ_i γ_i·sumcheck_claim_i),
    then b_combined gains Σ_j γ_pd_j·eq_ind_j and the target Σ_j γ_pd_j·value_j; the
    recursive Ligerito prover runs against (b_combined, target). γ order: the
    ring-switch γ's first (sampled inside prove_batched), then γ_pd after observing
    each packed-direct value. M=0 recovers the plain Ligerito open (open_batch_ligerito).
    `pdata` is the ligerito commit reused from the commit phase (no L0 re-encode).
    Returns the batch-opening proof and advanced transcript."""
    transcript = transcript.observe_label(b"flock-pcs-open-batch-v0")
    s_hat_vs, rs_eq_inds, sumcheck_claims, gammas, transcript = (
        ring_switch.prove_batched(z_packed, x_outers, transcript)
    )
    # Packed-direct: observe each claim's value, THEN sample the γ_pd (flock order).
    for pd in packed_direct:
        transcript = transcript.observe_label(b"flock-pcs-packed-direct-v0")
        transcript = transcript.observe_f128(pd.value)
    gammas_pd = []
    for _ in packed_direct:
        transcript, gamma = transcript.sample_f128()
        gammas_pd.append(gamma)

    b_combined, target = _combine_claims(rs_eq_inds, gammas, sumcheck_claims,
                                         packed_direct=packed_direct, gammas_pd=gammas_pd)
    # The Ligerito recursion runs in zorch (`zorch.pcs.ligerito`) via the flock
    # FS seam, reusing the commit-phase `pdata` directly. The ghash algebra rides
    # the dtype, so `mul` is not threaded.
    lig, transcript = zorch_ligerito.prove_flock_ligerito(
        config, pdata, b_combined, target, transcript
    )
    return BatchOpenProof(ring_switches=s_hat_vs, ligerito=lig), transcript


def prove_fast(z_packed: Array, m: int, k_log: int, k_skip: int,
               a0: Array, b0: Array, z_lincheck: bytes, statement_digest: bytes,
               cfg, circuit=None, domain: bytes = b"flock-test-v0") -> ProveFastResult:
    """Fused single-call R1CS prover on the Ligerito PCS, byte-identical to flock
    `prover::prove_fast_ligerito`. Flock's phases have protocol-specific dataflow
    and proof formats, so the composition is ordinary Python rather than a generic
    stage chain. `cfg` is the flock Ligerito config; `circuit` a `LincheckCircuit`
    for real hash R1CS (None uses the dense a0/b0 path — the identity gate).
    a = A·z, b = B·z; for the identity R1CS a = b = c = z. Returns the proof +
    claims."""
    transcript = flock_transcript(domain)
    root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, z_packed)
    transcript = bind_statement(transcript, statement_digest, root)

    bits = _unpack_bits(fnp.asarray(z_packed))
    zc, transcript = zerocheck.prove_packed(
        bits, bits, bits, m, transcript=transcript
    )

    inner_rest = k_log - k_skip
    x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, inner_rest)
    lp, transcript = lincheck.prove(
        z_lincheck,
        a0,
        b0,
        x_ab,
        m,
        k_log,
        k_skip,
        transcript=transcript,
        capture=True,
        circuit=circuit,
    )
    if lp.claim is None:
        raise ValueError("captured lincheck did not produce its opening claim")

    x_outer = zc.mlv_challenges[inner_rest:]
    ab_full = fnp.concatenate([lp.claim.r_inner_rest, x_outer], axis=0)
    # Split-then-rejoin to mirror Rust's QuirkyPoint / quirky_x_outer_full.
    c_full = fnp.concatenate([zc.r_rest[:inner_rest], zc.r_rest[inner_rest:]], axis=0)
    pcs_open_proof, _transcript = open_batch_ligerito(
        cfg, z_packed, pdata, [ab_full, c_full], transcript
    )

    return ProveFastResult(
        zerocheck=zc,
        lincheck=(lp.rounds, lp.z_partial),
        pcs_open=pcs_open_proof,
        claim_ab_value=lp.claim.w,
        claim_c_value=zc.final_c_eval,
    )
