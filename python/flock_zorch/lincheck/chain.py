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
import functools
from typing import Any

import numpy as np
import frx
import frx.numpy as fnp

from flock_zorch import ghash
from flock_zorch.sumcheck import build_eq
from flock_zorch.sumcheck.inf_product import prove_inf_product


@dataclass(frozen=True)
class PackedDirectClaim:
    """A packed-direct PCS claim: a ẑ-evaluation `value` at `point` (its eq_ind is
    `build_eq(point)`), combined into the batched open alongside the ring-switched
    claims."""

    point: Any
    value: Any

LOG_PACKING = ghash.LOG_PACKING  # 128 = 2^7 bits per packed F128 element


def prove_chain_shift(in_vals, out_vals, ch):
    """flock `chain::prove_chain_shift`. in_vals/out_vals: native ghash (2^n,)
    (already region-folded per instance). Threads τ, α, and the sumcheck
    challenges through the shared challenger `ch`. Returns (rounds [(e1,einf)],
    g_at_point, claims) where claims = {instance_point: ghash (n,), sel0: ghash
    scalar, value: ghash scalar} — device-resident for `assemble_chain_claim`."""
    in_vals = fnp.asarray(in_vals).reshape(-1)   # ghash [n_total]
    out_vals = fnp.asarray(out_vals).reshape(-1)
    n_total = in_vals.shape[0]
    assert out_vals.shape[0] == n_total and (n_total & (n_total - 1)) == 0
    n = int(n_total).bit_length() - 1

    # τ ∈ Fⁿ then α — both before the sumcheck (mirrored by the verifier).
    tau = ch.sample_f128(n)
    alpha = ch.sample_f128()
    eqtau = build_eq(tau)                       # eqtau[y] = eq(τ, y), ghash [n_total]

    # Weight table over (y, s₀), s₀ the HIGH bit (index y + s₀·N):
    #   W(y,0) = shift(τ,y) + α·eq(y,0ⁿ) = eqtau[y-1] (y≥1), α at y==0
    #   W(y,1) = eq(τ,y) = eqtau[y]
    w_lo = fnp.concatenate([alpha.reshape(1), eqtau[:n_total - 1]])  # [α, eqtau[0..N-2]]
    wt = fnp.concatenate([w_lo, eqtau])                             # [2·N]
    g = fnp.concatenate([in_vals, out_vals])                        # [In ‖ Out]

    # Product sumcheck Σ_{y,s₀} W·g over n+1 vars (round msg + fold == lincheck's).
    stacked = fnp.stack([wt, g])
    stacked, ch._t, msgs = prove_inf_product(stacked, ch._t, n + 1)
    rounds = [(e1, einf) for e1, einf, _ in msgs]

    # After n+1 folds g[0] = g(τ',s₀*). Build the point: full[d-1-k]=r_k (bit
    # d-1 = s₀, the HIGH bit), i.e. the fold challenges reversed — native ghash,
    # no host lift; τ' = full[:n], s₀* = full[n].
    value = stacked[1].reshape(())              # native ghash scalar
    full = fnp.stack([fnp.reshape(r, ()) for _, _, r in reversed(msgs)])
    claims = {"instance_point": full[:n], "sel0": full[n], "value": value}
    return rounds, value, claims


def assemble_chain_claim(tau_pos, claims, k_log, region_log):
    """flock `chain_common::assemble_chain_claim` — the packed-direct chain claim.
    point (LSB-first, length m−7) = [τ_pos, sel0, 0^high, instance_point],
    high = k_log − region_log − 1; value = claims.value — native ghash throughout:
    the PCS opens ẑ at this point (its eq_ind == build_eq(point), what
    build_eq_sparse computes), so it never needs to leave the device."""
    high = k_log - region_log - 1
    point = fnp.concatenate([
        fnp.reshape(tau_pos, (-1,)),
        fnp.reshape(claims["sel0"], (1,)),
        ghash.zeros(high),
        fnp.reshape(claims["instance_point"], (-1,)),
    ], axis=0)
    return PackedDirectClaim(point=point, value=claims["value"])


def fold_in_out(packed, k_log, tau_pos, input_byte_off, output_byte_off):
    """flock `chain_common::fold_in_out`. Collapse each instance's input/output
    slot to one F128: in_vals[i] = Σ_pos eq(τ_pos, pos)·ẑ_packed[(i, in_slot, pos)],
    likewise out_vals. `packed` is ẑ (length 2^(m-7), host uint64 lanes — the
    committed witness arrives from the wire); `tau_pos` is a native ghash [t]
    challenge draw, t = region_log − LOG_PACKING. Returns (in_vals, out_vals),
    each native ghash (2^n,)."""
    packed = np.asarray(packed, np.uint64).reshape(-1, 2)
    n_packed = 1 << tau_pos.shape[0]
    block_packed = (1 << k_log) >> LOG_PACKING
    in_base = (input_byte_off * 8) >> LOG_PACKING
    out_base = (output_byte_off * 8) >> LOG_PACKING
    assert packed.shape[0] % block_packed == 0
    n_inst = packed.shape[0] // block_packed
    pk = ghash.to_ghash(fnp.asarray(packed).reshape(n_inst, block_packed, 2))  # (n_inst, block_packed)
    return _fold_in_out_core(pk, tau_pos, in_base, out_base, n_packed)


@functools.partial(frx.jit, static_argnums=(2, 3, 4))
def _fold_in_out_core(pk, tau_pos, in_base, out_base, n_packed):
    """The device half of `fold_in_out`: build eq(τ) and the two eq-weighted region
    sums in one jitted kernel. `build_eq` is in-kernel (no `build_eq_fused`)."""
    eq_tau = build_eq(tau_pos)                                   # (n_packed,) ghash
    in_reg = pk[:, in_base:in_base + n_packed]                   # (n_inst, n_packed)
    out_reg = pk[:, out_base:out_base + n_packed]
    in_vals = fnp.sum(in_reg * eq_tau[None], axis=1)            # (n_inst,) ghash
    out_vals = fnp.sum(out_reg * eq_tau[None], axis=1)
    return in_vals, out_vals
