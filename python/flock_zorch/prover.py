"""flock's fused R1CS prover (`prover::prove` / `prove_fast_core`), authored in
jax — byte-identical to flock-core. Chains the byte-identical phases on ONE
shared SHA-256 challenger with device-resident state (no per-phase host
re-transfer): commit → bind_statement → zerocheck → lincheck → batched PCS open.

This is the honest single-call e2e measurement (vs the standalone-phase sum in
e2e_gpu_bench) and removes the witness transfer (a=A·z, b=B·z are device-
resident). Gated by `testing/e2e_oracle_test.py` against flock `prover::prove`.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from flock_zorch import field, ring_switch, basefold, pcs_open, pcs_commit, zerocheck, lincheck, ligerito
from flock_zorch.sumcheck import build_eq
from flock_zorch.challenger import Challenger  # noqa: F401  (re-exported for callers)


@jax.jit
def _unpack_bits_dev(z_packed):
    """Packed F128 witness [2^(m-7),2] -> device bit witness [2^m] uint8 (LSB-first
    within each 128-bit element), on device so a=b=c=z stays device-resident."""
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


def open_batch(z_packed, codeword, init_tree, x_outers, k_code, log_inv_rate,
               log_batch_size, ch, mul=field.mul, use_host_sha: bool = False) -> dict:
    """Batched dual-claim PCS open — byte-identical to flock
    `pcs::open_batch_padded_with_precomputed_s_hat_v` (BatchOpeningProof =
    {ring_switches, basefold}). Each x_outers[i] = quirky_x_outer_full(claim.point)
    = x_inner_rest ++ x_outer. N ring-switch reductions are γ-combined into ONE
    BaseFold: b_combined = Σ_i γ_i·rs_eq_ind_i, run on a=z_packed. (round0_prime
    precompute is byte-equivalent to recomputing the round-0 message, so the
    existing basefold.prove suffices; target_combined doesn't affect proof bytes.)"""
    ch.observe_label(b"flock-pcs-open-batch-v0")
    s_hat_vs, rs_eq_inds, _scs, _gammas = ring_switch.prove_batched(z_packed, x_outers, ch, mul=mul)
    b_combined = jnp.asarray(rs_eq_inds[0])
    for r in rs_eq_inds[1:]:
        b_combined = field.add(b_combined, jnp.asarray(r))   # γ already baked in
    b_combined = np.asarray(b_combined)
    n_queries = pcs_open.default_fri_queries(log_inv_rate)
    bf = basefold.prove(z_packed, b_combined, codeword, init_tree, k_code,
                        log_inv_rate, log_batch_size, n_queries, ch, mul=mul,
                        use_host_sha=use_host_sha)
    return {"ring_switches": s_hat_vs, "basefold": bf}


def open_batch_ligerito(config, z_packed, codeword, init_tree, x_outers, ch,
                        mul=field.mul, use_host_sha=False) -> dict:
    """Batched dual-claim PCS open with the LIGERITO backend (flock
    `open_batch_mixed_ligerito_with_precomputed_s_hat_v`) — the headline path.
    Same combine as open_batch (Σ_i γ_i·rs_eq_ind_i → b_combined) but drives the
    recursive Ligerito prover instead of a single BaseFold. target_combined =
    Σ_i γ_i·sumcheck_claim_i. Returns {ring_switches, ligerito: LigeritoProof}."""
    ch.observe_label(b"flock-pcs-open-batch-v0")
    s_hat_vs, rs_eq_inds, sumcheck_claims, gammas = ring_switch.prove_batched(z_packed, x_outers, ch, mul=mul)
    b_combined = jnp.asarray(rs_eq_inds[0])
    for r in rs_eq_inds[1:]:
        b_combined = field.add(b_combined, jnp.asarray(r))
    target = jnp.zeros(2, jnp.uint64)
    for g, sc in zip(gammas, sumcheck_claims):
        target = field.add(target, mul(jnp.asarray(g), jnp.asarray(sc)))
    lig = ligerito.recursive_prover_with_basis(
        config, np.asarray(z_packed), np.asarray(b_combined), np.asarray(target),
        codeword, init_tree, ch, mul=mul, use_host_sha=use_host_sha)
    return {"ring_switches": s_hat_vs, "ligerito": lig}


def open_batch_mixed_ligerito(config, z_packed, codeword, init_tree, x_outers, packed_direct,
                              ch, mul=field.mul, use_host_sha=False) -> dict:
    """Mixed batched open (flock `open_batch_mixed_ligerito_with_precomputed_s_hat_v`)
    — the HASH-CHAIN open. Combines N ring-switched claims (x_outers, e.g. ab+c)
    with M packed-direct claims (the chain claim: a direct ẑ-evaluation at a point,
    eq_ind = build_eq(point) == build_eq_sparse(point)). Same recursive Ligerito
    driver as open_batch_ligerito; b_combined just gains Σ_j γ_pd_j·eq_ind_j and the
    target gains Σ_j γ_pd_j·value_j. γ order: the ring-switch γ's first (sampled
    inside prove_batched), then γ_pd after observing each packed-direct value."""
    ch.observe_label(b"flock-pcs-open-batch-v0")
    s_hat_vs, rs_eq_inds, sumcheck_claims, gammas = ring_switch.prove_batched(z_packed, x_outers, ch, mul=mul)
    # Packed-direct: observe each claim's value, THEN sample the γ_pd (flock order).
    for pd in packed_direct:
        ch.observe_label(b"flock-pcs-packed-direct-v0")
        ch.observe_f128(pd["value"])
    gammas_pd = [ch.sample_f128() for _ in packed_direct]

    b_combined = jnp.asarray(rs_eq_inds[0])           # γ_rs already baked in
    for r in rs_eq_inds[1:]:
        b_combined = field.add(b_combined, jnp.asarray(r))
    target = jnp.zeros(2, jnp.uint64)
    for g, sc in zip(gammas, sumcheck_claims):
        target = field.add(target, mul(jnp.asarray(g), jnp.asarray(sc)))
    for pd, g in zip(packed_direct, gammas_pd):
        eq_pd = build_eq(jnp.asarray(pd["point"]), mul=mul)   # length L = 2^(m-7)
        gj = jnp.asarray(g)
        b_combined = field.add(b_combined, mul(gj, eq_pd))
        target = field.add(target, mul(gj, jnp.asarray(pd["value"])))

    lig = ligerito.recursive_prover_with_basis(
        config, np.asarray(z_packed), np.asarray(b_combined), np.asarray(target),
        codeword, init_tree, ch, mul=mul, use_host_sha=use_host_sha)
    return {"ring_switches": s_hat_vs, "ligerito": lig}


def prove_fast(z_packed, m, k_log, k_skip, useful_bits, a0, b0, z_lincheck, statement_digest,
               log_inv_rate=1, log_batch_size=5, domain=b"flock-test-v0", mul=field.mul,
               use_host_sha=False) -> dict:
    """Fused single-call R1CS prover (identity-C path: c = z), byte-identical to
    flock `prover::prove`. Keeps witness/codeword device-resident across all phases
    on ONE shared challenger (no per-phase host re-transfer): commit → bind →
    zerocheck → lincheck → batched dual-claim open. a = A·z, b = B·z; for the
    identity R1CS a = b = c = z (the gated path). Returns the proof dict + claims."""
    k_code = (m - 7 - log_batch_size) + log_inv_rate
    inner_rest = k_log - k_skip

    root, codeword, tree = pcs_commit.commit(z_packed, m, log_inv_rate, log_batch_size, mul, use_host_sha)
    ch = Challenger(domain)
    bind_statement(ch, statement_digest, root)

    bits = _unpack_bits_dev(jnp.asarray(z_packed))           # a = b = c = z, device-resident
    zc = zerocheck.prove_packed(bits, bits, bits, m, mul=mul, ch=ch)

    x_ab = {"z_skip": zc["z"],
            "x_inner_rest": zc["mlv_challenges"][:inner_rest],
            "x_outer": zc["mlv_challenges"][inner_rest:]}
    lc_rounds, lc_zp, lc_claim, _z_vec_pre = lincheck.prove(
        z_lincheck, a0, b0, x_ab, m, k_log, k_skip, mul=mul, ch=ch, capture=True)

    ab_full = np.concatenate([lc_claim["r_inner_rest"], x_ab["x_outer"]], axis=0)
    c_full = np.concatenate([zc["r_rest"][:inner_rest], zc["r_rest"][inner_rest:]], axis=0)
    pcs_open = open_batch(z_packed, codeword, tree, [ab_full, c_full], k_code,
                          log_inv_rate, log_batch_size, ch, mul=mul, use_host_sha=use_host_sha)

    return {"zerocheck": zc, "lincheck": (lc_rounds, lc_zp), "pcs_open": pcs_open,
            "claim_ab_value": lc_claim["w"], "claim_c_value": zc["final_c_eval"]}
