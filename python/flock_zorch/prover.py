"""flock's fused R1CS prover (`prover::prove` / `prove_fast_core`), authored in
frx — byte-identical to flock-core. A zorch `ProveChain` of Stages threading ONE
shared SHA-256 challenger with device-resident state (no per-phase host
re-transfer): commit+bind → zerocheck → lincheck → batched PCS open (see
`prove_fast`).

a = A·z, b = B·z are kept device-resident across the phases (no per-phase witness
re-transfer). Gated by `testing/e2e_oracle_test.py` against flock `prover::prove`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import frx
import frx.numpy as jnp
from frx import Array

from flock_zorch import field, zerocheck, lincheck
from flock_zorch.pcs import ring_switch, basefold, fri, ligerito as zorch_ligerito
from flock_zorch.sumcheck import build_eq_g
from flock_zorch.challenger import Challenger  # noqa: F401  (re-exported for callers)
from flock_zorch.pcs import FlockPcsProver
from zorch.round import ProveChain, Stage


@dataclass(frozen=True)
class BatchOpenProof:
    """Batched dual-claim PCS open (flock BatchOpeningProof): the per-claim
    ring-switch reductions plus the combined low-degree open — `basefold` for the
    BaseFold backend or `ligerito` for the Ligerito backend (exactly one set)."""

    ring_switches: Any
    basefold: Any = None
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
def _unpack_bits_dev(z_packed):
    """Packed F128 witness [2^(m-7),2] -> device bit witness [2^m] uint8 (LSB-first
    within each 128-bit element), on device so a=b=c=z stays device-resident. The
    inverse of pcs_commit.pack_witness (flock reads the packed ẑ back to bits the
    same way)."""
    bitidx = jnp.arange(64, dtype=jnp.uint64)
    lo = ((z_packed[:, 0:1] >> bitidx) & jnp.uint64(1)).astype(jnp.uint8)
    hi = ((z_packed[:, 1:2] >> bitidx) & jnp.uint64(1)).astype(jnp.uint8)
    return jnp.concatenate([lo, hi], axis=1).reshape(-1)


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


def _combine_claims(rs_eq_inds, gammas, sumcheck_claims, packed_direct=(), gammas_pd=()):
    """γ-combine the batched ring-switch claims (+ optional packed-direct claims) into
    the single (b_combined, target) the BaseFold/Ligerito open runs against. The
    ring-switch γ's are already baked into each rs_eq_ind by prove_batched, so b is
    their XOR-sum; target = Σ γ_i·sumcheck_claim_i. Packed-direct claims add
    γ_pd_j·eq(point_j) to b and γ_pd_j·value_j to target. NB: all observe/sample stay
    at the call sites — this is pure arithmetic, so it cannot perturb the transcript."""
    b_combined = rs_eq_inds[0]                                     # native ghash [2^L]
    for r in rs_eq_inds[1:]:
        b_combined = b_combined + r                                # γ_rs already baked in
    target = field.to_ghash(jnp.zeros(2, jnp.uint64))              # ghash scalar zero
    for g, sc in zip(gammas, sumcheck_claims):                     # g native ghash
        target = target + g * field.to_ghash(jnp.asarray(sc))
    for pd, g in zip(packed_direct, gammas_pd):                    # g native ghash
        eq_pd = build_eq_g(field.to_ghash(jnp.asarray(pd.point)))   # length L = 2^(m-7)
        gj = g
        b_combined = b_combined + gj * eq_pd
        target = target + gj * field.to_ghash(jnp.asarray(pd.value))
    return field.from_ghash(b_combined), field.from_ghash(target)


def open_batch(z_packed, codeword, init_tree, x_outers, k_code, log_inv_rate,
               log_batch_size, ch) -> BatchOpenProof:
    """Batched dual-claim PCS open — byte-identical to flock
    `pcs::open_batch_padded_with_precomputed_s_hat_v` (BatchOpeningProof =
    {ring_switches, basefold}). Each x_outers[i] = quirky_x_outer_full(claim.point)
    = x_inner_rest ++ x_outer. N ring-switch reductions are γ-combined into ONE
    BaseFold: b_combined = Σ_i γ_i·rs_eq_ind_i, run on a=z_packed. (round0_prime
    precompute is byte-equivalent to recomputing the round-0 message, so the
    existing basefold.prove suffices; target_combined doesn't affect proof bytes.)"""
    ch.observe_label(b"flock-pcs-open-batch-v0")
    s_hat_vs, rs_eq_inds, sumcheck_claims, gammas = ring_switch.prove_batched(z_packed, x_outers, ch)
    b_combined, _target = _combine_claims(rs_eq_inds, gammas, sumcheck_claims)  # BaseFold ignores target
    b_combined = np.asarray(b_combined)
    n_queries = fri.default_fri_queries(log_inv_rate)
    bf = basefold.prove(z_packed, b_combined, codeword, init_tree, k_code,
                        log_inv_rate, log_batch_size, n_queries, ch)
    return BatchOpenProof(ring_switches=s_hat_vs, basefold=bf)


def open_batch_ligerito(config, z_packed, pdata, x_outers, ch) -> BatchOpenProof:
    """Batched dual-claim PCS open with the LIGERITO backend — the headline path.
    The no-packed-direct case of `open_batch_mixed_ligerito`: N ring-switched
    claims (x_outers, e.g. ab+c), zero direct ẑ-evaluation claims. `pdata` is the
    ligerito commit from `zorch_ligerito.commit_flock_ligerito`. Returns
    {ring_switches, ligerito: LigeritoProof}."""
    return open_batch_mixed_ligerito(config, z_packed, pdata, x_outers, (), ch)


def open_batch_mixed_ligerito(config, z_packed, pdata, x_outers, packed_direct,
                              ch) -> BatchOpenProof:
    """Mixed batched open (flock `open_batch_mixed_ligerito_with_precomputed_s_hat_v`)
    — the HASH-CHAIN open, and the general Ligerito open. Combines N ring-switched
    claims (x_outers, e.g. ab+c) with M packed-direct claims (the chain claim: a
    direct ẑ-evaluation at a point, eq_ind = build_eq(point) == build_eq_sparse(point)).
    The combine is Σ_i γ_i·rs_eq_ind_i → b_combined (target Σ_i γ_i·sumcheck_claim_i),
    then b_combined gains Σ_j γ_pd_j·eq_ind_j and the target Σ_j γ_pd_j·value_j; the
    recursive Ligerito prover runs against (b_combined, target). γ order: the
    ring-switch γ's first (sampled inside prove_batched), then γ_pd after observing
    each packed-direct value. M=0 recovers the plain Ligerito open (open_batch_ligerito).
    `pdata` is the ligerito commit reused from the commit phase (no L0 re-encode)."""
    ch.observe_label(b"flock-pcs-open-batch-v0")
    s_hat_vs, rs_eq_inds, sumcheck_claims, gammas = ring_switch.prove_batched(z_packed, x_outers, ch)
    # Packed-direct: observe each claim's value, THEN sample the γ_pd (flock order).
    for pd in packed_direct:
        ch.observe_label(b"flock-pcs-packed-direct-v0")
        ch.observe_f128(pd.value)
    gammas_pd = [ch.sample_f128_g() for _ in packed_direct]  # native ghash

    b_combined, target = _combine_claims(rs_eq_inds, gammas, sumcheck_claims,
                                         packed_direct=packed_direct, gammas_pd=gammas_pd)
    # The Ligerito recursion runs in zorch (`zorch.pcs.ligerito`) via the flock
    # FS seam, reusing the commit-phase `pdata` directly. The ghash algebra rides
    # the dtype, so `mul` is not threaded.
    lig = zorch_ligerito.prove_flock_ligerito(config, pdata, b_combined, target, ch)
    return BatchOpenProof(ring_switches=s_hat_vs, ligerito=lig)


@dataclass(frozen=True)
class _ProveCarry:
    """State threaded between prove_fast's stages — only what a later stage reads
    from an earlier one. Static config (shapes, rates) lives on the stage
    instances, per zorch's `Round`-interface convention (cf. sp1-zorch ShardCarry).
    The `None` fields are the per-stage outputs, written via `replace`."""

    z_packed: Array           # witness ẑ, device-resident across stages
    statement_digest: bytes   # R1CS instance digest (bound by _CommitStage)
    z_lincheck: bytes         # lincheck witness bytes
    a0: Array                 # lincheck A matrix (dense)
    b0: Array                 # lincheck B matrix (dense)
    codeword: np.ndarray | None = None  # ← _CommitStage; read by _PcsOpenStage
    tree: np.ndarray | None = None      # ← _CommitStage; read by _PcsOpenStage
    zc: zerocheck.ZerocheckProof | None = None    # ← _ZerocheckStage; read by lincheck + open + assembly
    lc_claim: lincheck.LincheckClaim | None = None  # ← _LincheckStage; read by open + assembly


class _CommitStage(Stage):
    """Commit ẑ through the `FlockPcsProver` seam, then bind the transcript to the
    statement (flock `bind_statement`): the trace-commit Stage (commit + preamble
    absorb). Message = the root."""

    def __init__(self, pcs: FlockPcsProver):
        self._pcs = pcs

    def __call__(self, carry, transcript):
        root, data = self._pcs.commit([carry.z_packed])
        bind_statement(transcript, carry.statement_digest, root)
        return replace(carry, codeword=data.codeword, tree=data.tree), transcript, root


class _ZerocheckStage(Stage):
    """R1CS zerocheck on the identity witness (a = b = c = ẑ). Message = the
    zerocheck proof/claim dict, also threaded onto the carry for later stages."""

    def __init__(self, m):
        self._m = m

    def __call__(self, carry, transcript):
        bits = _unpack_bits_dev(jnp.asarray(carry.z_packed))   # device-resident
        zc = zerocheck.prove_packed(bits, bits, bits, self._m, ch=transcript)
        return replace(carry, zc=zc), transcript, zc


class _LincheckStage(Stage):
    """Lincheck reducing a = A·z, b = B·z to the ab evaluation claim at the
    zerocheck challenge point. Message = (rounds, z_partial); writes the ab claim
    onto the carry."""

    def __init__(self, m, k_log, k_skip):
        self._m, self._k_log, self._k_skip = m, k_log, k_skip

    def __call__(self, carry, transcript):
        if carry.zc is None:
            raise ValueError("lincheck needs the zerocheck output on the carry; "
                             "sequence a _ZerocheckStage before this stage")
        inner_rest = self._k_log - self._k_skip
        zc = carry.zc
        x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, inner_rest)
        lp = lincheck.prove(
            carry.z_lincheck, carry.a0, carry.b0, x_ab, self._m,
            self._k_log, self._k_skip, ch=transcript, capture=True)
        return replace(carry, lc_claim=lp.claim), transcript, (lp.rounds, lp.z_partial)


class _PcsOpenStage(Stage):
    """Batched dual-claim PCS open of the ab + c claims — the final stage. ab
    point = lincheck r_inner_rest ++ zerocheck x_outer; c point = the zerocheck
    r_rest. Message = the BatchOpeningProof dict."""

    def __init__(self, pcs: FlockPcsProver, k_log, k_skip):
        self._pcs, self._k_log, self._k_skip = pcs, k_log, k_skip

    def __call__(self, carry, transcript):
        if (carry.zc is None or carry.lc_claim is None
                or carry.codeword is None or carry.tree is None):
            raise ValueError("the PCS open needs the commit codeword/tree plus the "
                             "zerocheck and lincheck outputs on the carry; sequence "
                             "the commit, zerocheck, and lincheck Stages before it")
        inner_rest = self._k_log - self._k_skip
        zc, lc_claim = carry.zc, carry.lc_claim
        x_outer = zc.mlv_challenges[inner_rest:]
        ab_full = np.concatenate([lc_claim.r_inner_rest, x_outer], axis=0)
        # c_full split-then-rejoined (not just zc.r_rest) to mirror Rust's
        # QuirkyPoint / quirky_x_outer_full.
        c_full = np.concatenate([zc.r_rest[:inner_rest], zc.r_rest[inner_rest:]], axis=0)
        pcs = self._pcs
        pcs_open_proof = open_batch(
            carry.z_packed, carry.codeword, carry.tree, [ab_full, c_full], pcs.k_code,
            pcs.log_inv_rate, pcs.log_batch_size, transcript)
        return carry, transcript, pcs_open_proof


def prove_fast(z_packed: Array, m: int, k_log: int, k_skip: int, useful_bits: int,
               a0: Array, b0: Array, z_lincheck: bytes, statement_digest: bytes,
               log_inv_rate: int = 1, log_batch_size: int = 5,
               domain: bytes = b"flock-test-v0") -> ProveFastResult:
    """Fused single-call R1CS prover (identity-C path: c = z), byte-identical to
    flock `prover::prove`. A zorch `ProveChain` of Stages threading one shared
    challenger + a `_ProveCarry` (no per-phase host re-transfer): commit+bind →
    zerocheck → lincheck → batched dual-claim open. a = A·z, b = B·z; for the
    identity R1CS a = b = c = z (the gated path). Returns the proof + claims."""
    pcs = FlockPcsProver(m, log_inv_rate, log_batch_size)

    ch = Challenger(domain)
    carry = _ProveCarry(z_packed=z_packed, statement_digest=statement_digest,
                        z_lincheck=z_lincheck, a0=a0, b0=b0)
    carry, _ch, msgs = ProveChain([
        _CommitStage(pcs),
        _ZerocheckStage(m),
        _LincheckStage(m, k_log, k_skip),
        _PcsOpenStage(pcs, k_log, k_skip),
    ])(carry, ch)
    _root, zc, (lc_rounds, lc_zp), pcs_open_proof = msgs

    return ProveFastResult(zerocheck=zc, lincheck=(lc_rounds, lc_zp), pcs_open=pcs_open_proof,
                           claim_ab_value=carry.lc_claim.w, claim_c_value=zc.final_c_eval)
