"""Hash-chain "shift" argument — Python port of flock's prover-side chain glue
(`flock-prover/src/chain.rs` `prove_chain_shift` + `r1cs_hashes/chain_common.rs`
`fold_in_out`).

The chain protocol glues 2^n committed hash instances into a sequential chain
`x_{i+1}=h(x_i)` with public endpoints, via:
  1. **region fold** (`fold_in_out`): collapse each instance's state_0/state_24
     slot to one F128 using the verifier's τ_pos — `In(i)`, `Out(i)`.
  2. **shift sumcheck** (`prove_chain_shift`): one (n+1)-round product sumcheck
     over (y, s₀) with weight W(y,s₀) = shift(τ,y)·(1+s₀) + eq(τ,y)·s₀ +
     α·eq(y,0ⁿ)·(1+s₀), reducing the glue + both endpoints to a SINGLE ẑ-eval
     claim `g(τ',s₀*)`.

The round message + fold are the shared ∞-product round
(`sumcheck.inf_product`, the same wire lincheck's rounds ride).
`assemble_chain_claim` then packs the returned claim into the point the mixed
packed-direct PCS open consumes (the `ChainProof` assembly).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import frx.numpy as jnp

from flock_zorch import field
from flock_zorch.sumcheck import build_eq, build_eq_g
from flock_zorch.sumcheck.inf_product import prove_inf_product


@dataclass(frozen=True)
class PackedDirectClaim:
    """A packed-direct PCS claim: a ẑ-evaluation `value` at `point` (its eq_ind is
    `build_eq(point)`), combined into the batched open alongside the ring-switched
    claims."""

    point: Any
    value: Any

LOG_PACKING = field.LOG_PACKING  # 128 = 2^7 bits per packed F128 element


def prove_chain_shift(in_vals, out_vals, ch):
    """flock `chain::prove_chain_shift`. in_vals/out_vals: (2^n, 2) F128 (already
    region-folded per instance). Threads τ, α, and the sumcheck challenges through
    the shared challenger `ch`. Returns (rounds [(e1,einf)], g_at_point, claims)
    where claims = {instance_point (n,2), sel0 (2,), value (2,)}."""
    in_vals = np.asarray(in_vals, np.uint64).reshape(-1, 2)
    out_vals = np.asarray(out_vals, np.uint64).reshape(-1, 2)
    n_total = in_vals.shape[0]
    assert out_vals.shape[0] == n_total and (n_total & (n_total - 1)) == 0
    n = int(n_total).bit_length() - 1

    # τ ∈ Fⁿ then α — both before the sumcheck (mirrored by the verifier).
    tau = np.asarray(ch.sample_f128_vec(n)).reshape(n, 2)
    alpha = np.asarray(ch.sample_f128()).reshape(2)
    eqtau = np.asarray(build_eq(jnp.asarray(tau))).reshape(n_total, 2)  # eqtau[y]=eq(τ,y)

    # Weight table over (y, s₀), s₀ the HIGH bit (index y + s₀·N):
    #   W(y,0) = shift(τ,y) + α·eq(y,0ⁿ) = eqtau[y-1] (y≥1) + α·[y==0]
    #   W(y,1) = eq(τ,y) = eqtau[y]
    wt = np.zeros((2 * n_total, 2), np.uint64)
    wt[1:n_total] = eqtau[:n_total - 1]
    wt[0] ^= alpha
    wt[n_total:] = eqtau
    g = np.concatenate([in_vals, out_vals], axis=0)  # [In ‖ Out]

    # Product sumcheck Σ_{y,s₀} W·g over n+1 vars (round msg + fold == lincheck's).
    stacked = jnp.stack([field.to_ghash(jnp.asarray(wt)),
                         field.to_ghash(jnp.asarray(g))])
    stacked, ch._t, msgs = prove_inf_product(stacked, ch._t, n + 1)
    # Messages ride ghash out of the fused rounds; one host materialization.
    e1s_g, einfs_g, r_g = zip(*msgs)
    e1s = field.from_ghash_host(jnp.stack(e1s_g))
    einfs = field.from_ghash_host(jnp.stack(einfs_g))
    r_pts = list(field.from_ghash_host(jnp.stack(r_g)).reshape(-1, 2))
    rounds = list(zip(e1s, einfs))

    # After n+1 folds g[0] = g(τ',s₀*). Build the point: full[d-1-k]=r_pts[k]
    # (bit d-1 = s₀, the HIGH bit); τ' = full[:n], s₀* = full[n].
    value = field.from_ghash_host(stacked[1]).reshape(2)
    d = n + 1
    full = np.zeros((d, 2), np.uint64)
    for k, r in enumerate(r_pts):
        full[d - 1 - k] = r
    claims = {"instance_point": full[:n].copy(), "sel0": full[n].copy(), "value": value}
    return rounds, value, claims


def assemble_chain_claim(tau_pos, claims, k_log, region_log):
    """flock `chain_common::assemble_chain_claim` — the packed-direct chain claim.
    point (LSB-first, length m−7) = [τ_pos, sel0, 0^high, instance_point],
    high = k_log − region_log − 1; value = claims.value. The PCS opens ẑ at this
    point (its eq_ind == build_eq(point), what build_eq_sparse computes)."""
    tau_pos = np.asarray(tau_pos, np.uint64).reshape(-1, 2)
    high = k_log - region_log - 1
    point = np.concatenate([
        tau_pos,
        np.asarray(claims["sel0"], np.uint64).reshape(1, 2),
        np.zeros((high, 2), np.uint64),
        np.asarray(claims["instance_point"], np.uint64).reshape(-1, 2),
    ], axis=0)
    return PackedDirectClaim(point=point,
                             value=np.asarray(claims["value"], np.uint64).reshape(2))


def fold_in_out(packed, k_log, tau_pos, input_byte_off, output_byte_off):
    """flock `chain_common::fold_in_out`. Collapse each instance's input/output
    slot to one F128: in_vals[i] = Σ_pos eq(τ_pos, pos)·ẑ_packed[(i, in_slot, pos)],
    likewise out_vals. `packed` is ẑ (length 2^(m-7)); `tau_pos` length =
    region_log − LOG_PACKING. Returns (in_vals, out_vals), each (2^n, 2)."""
    packed = np.asarray(packed, np.uint64).reshape(-1, 2)
    tau_pos = np.asarray(tau_pos, np.uint64).reshape(-1, 2)
    n_packed = 1 << tau_pos.shape[0]
    block_packed = (1 << k_log) >> LOG_PACKING
    in_base = (input_byte_off * 8) >> LOG_PACKING
    out_base = (output_byte_off * 8) >> LOG_PACKING
    assert packed.shape[0] % block_packed == 0
    n_inst = packed.shape[0] // block_packed
    eq_tau = build_eq_g(field.to_ghash(jnp.asarray(tau_pos)))  # (n_packed,)

    pk = field.to_ghash(jnp.asarray(packed).reshape(n_inst, block_packed, 2))  # (n_inst, block_packed)
    in_reg = pk[:, in_base:in_base + n_packed]                   # (n_inst, n_packed)
    out_reg = pk[:, out_base:out_base + n_packed]
    in_vals = jnp.sum(in_reg * eq_tau[None], axis=1)            # (n_inst,)
    out_vals = jnp.sum(out_reg * eq_tau[None], axis=1)
    return field.from_ghash_host(in_vals), field.from_ghash_host(out_vals)
