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
import jax.numpy as jnp

from flock_zorch import field, ring_switch, basefold, pcs_open
from flock_zorch.challenger import Challenger  # noqa: F401  (re-exported for callers)


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
