"""Hash-chain "shift" argument — Python port of flock's prover-side chain glue
(`flock-prover/src/chain.rs` `prove_chain_shift` + `r1cs_hashes/chain_common.rs`
`fold_in_out`). Task #14, M4a.

The chain protocol glues 2^n committed hash instances into a sequential chain
`x_{i+1}=h(x_i)` with public endpoints, via:
  1. **region fold** (`fold_in_out`): collapse each instance's state_0/state_24
     slot to one F128 using the verifier's τ_pos — `In(i)`, `Out(i)`.
  2. **shift sumcheck** (`prove_chain_shift`): one (n+1)-round product sumcheck
     over (y, s₀) with weight W(y,s₀) = shift(τ,y)·(1+s₀) + eq(τ,y)·s₀ +
     α·eq(y,0ⁿ)·(1+s₀), reducing the glue + both endpoints to a SINGLE ẑ-eval
     claim `g(τ',s₀*)`.

The round message + fold are identical to the existing product sumcheck
(`lincheck._round_eval` / `_bind_top`), so those are reused. The remaining chain
milestone is the mixed packed-direct PCS open that consumes the returned claim.
"""

import numpy as np
import jax.numpy as jnp

from flock_zorch import field
from flock_zorch.sumcheck import build_eq, _xor_reduce
from flock_zorch.lincheck import _round_eval, _bind_top

LOG_PACKING = 7  # 128 = 2^7 bits per packed F128 element


def prove_chain_shift(in_vals, out_vals, ch, mul=field.mul):
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
    eqtau = np.asarray(build_eq(jnp.asarray(tau), mul=mul)).reshape(n_total, 2)  # eqtau[y]=eq(τ,y)

    # Weight table over (y, s₀), s₀ the HIGH bit (index y + s₀·N):
    #   W(y,0) = shift(τ,y) + α·eq(y,0ⁿ) = eqtau[y-1] (y≥1) + α·[y==0]
    #   W(y,1) = eq(τ,y) = eqtau[y]
    wt = np.zeros((2 * n_total, 2), np.uint64)
    wt[1:n_total] = eqtau[:n_total - 1]
    wt[0] ^= alpha
    wt[n_total:] = eqtau
    g = np.concatenate([in_vals, out_vals], axis=0)  # [In ‖ Out]

    # Product sumcheck Σ_{y,s₀} W·g over n+1 vars (round msg + fold == lincheck's).
    wt = jnp.asarray(wt); g = jnp.asarray(g)
    rounds, r_pts = [], []
    for _ in range(n + 1):
        e1, einf = _round_eval(wt, g, mul)
        ch.observe_f128(np.asarray(e1)); ch.observe_f128(np.asarray(einf))
        r = jnp.asarray(ch.sample_f128())
        rounds.append((np.asarray(e1), np.asarray(einf)))
        r_pts.append(np.asarray(r).reshape(2))
        wt = _bind_top(wt, r, mul)
        g = _bind_top(g, r, mul)

    # After n+1 folds g[0] = g(τ',s₀*). Build the point: full[d-1-k]=r_pts[k]
    # (bit d-1 = s₀, the HIGH bit); τ' = full[:n], s₀* = full[n].
    value = np.asarray(g[0]).reshape(2)
    d = n + 1
    full = np.zeros((d, 2), np.uint64)
    for k, r in enumerate(r_pts):
        full[d - 1 - k] = r
    claims = {"instance_point": full[:n].copy(), "sel0": full[n].copy(), "value": value}
    return rounds, value, claims


def fold_in_out(packed, k_log, tau_pos, input_byte_off, output_byte_off, mul=field.mul):
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
    eq_tau = build_eq(jnp.asarray(tau_pos), mul=mul)             # (n_packed, 2)

    pk = jnp.asarray(packed).reshape(n_inst, block_packed, 2)
    in_reg = pk[:, in_base:in_base + n_packed, :]                # (n_inst, n_packed, 2)
    out_reg = pk[:, out_base:out_base + n_packed, :]
    in_vals = _xor_reduce(mul(in_reg, eq_tau[None]), axis=1)     # (n_inst, 2)
    out_vals = _xor_reduce(mul(out_reg, eq_tau[None]), axis=1)
    return np.asarray(in_vals), np.asarray(out_vals)
