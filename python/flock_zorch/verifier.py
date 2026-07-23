# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""R1CS proof verifier (dense identity path).

`verify_core` replays bind → zerocheck → lincheck and rebuilds the ab/c z-claims;
`verify_claims_ligerito` reduces each through ring-switch, γ-combines to one
transparent basis, and hands it to zorch's Ligerito verifier. `ok` ANDs every
check. Native `binary_field_ghash`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp

from flock_zorch import ghash, sumcheck
from flock_zorch.lincheck import verifier as lincheck_verifier
from flock_zorch.lincheck.prover import AbClaimPoint, LincheckProof
from flock_zorch.pcs import ligerito as zorch_ligerito
from flock_zorch.pcs import ring_switch
from flock_zorch.prover import bind_statement
from flock_zorch.zerocheck import verifier as zerocheck_verifier
from zorch.pcs import ring_switch as zrs


@dataclass(frozen=True)
class _ZClaim:
    """A ẑ-evaluation claim: the point (z_skip skip-scalar ++ x_full outer coords)
    and its value — the shape the batched PCS open verifies."""

    z_skip: Any
    x_full: Any
    value: Any


def verify_core(cfg, root, statement, zc_proof, lc_proof, a0, b0, m, k_log, k_skip, ch):
    """Replay bind → zerocheck → lincheck; return `(ab, c, ok)` — the two z-claims
    the PCS open is checked against."""
    bind_statement(ch, statement, root)
    zc, _, ok1 = zerocheck_verifier.verify(m, zc_proof, ch)
    ir = k_log - k_skip
    x_ab = AbClaimPoint(z_skip=zc.z, x_inner_rest=zc.mlv_challenges[:ir],
                        x_outer=zc.mlv_challenges[ir:])
    lc, _, ok2 = lincheck_verifier.verify(m, k_log, k_skip, a0, b0, x_ab,
                                          zc.a_eval, zc.b_eval, lc_proof, ch)
    ab = _ZClaim(z_skip=lc.r_inner_skip,
                 x_full=fnp.concatenate([lc.r_inner_rest, x_ab.x_outer]), value=lc.w)
    c = _ZClaim(z_skip=zc.z,
                x_full=fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]]), value=zc.c_eval)
    return ab, c, ok1 & ok2


def verify_claims_ligerito(cfg, root, claims, pcs_open, ch):
    """Ring-switch reduce each z-claim, γ-combine to one (b_combined, target), and
    verify the single Ligerito opening. No packed-direct claims (identity R1CS)."""
    ch.observe_label(b"flock-pcs-open-batch-v0")
    reduced = []
    ok = True
    for claim, s_hat_v in zip(claims, pcs_open.ring_switches):
        sc, eq_r_dprime, ok_i = ring_switch.verify(
            claim.value, claim.z_skip, claim.x_full, s_hat_v, ch)
        reduced.append((sc, eq_r_dprime, claim.x_full))
        ok = ok & ok_i
    gammas = [ch.sample_f128() for _ in claims]

    target = ghash.to_ghash(fnp.zeros(2, fnp.uint64))
    b_combined = None
    for (sc, eq_r_dprime, x_full), g in zip(reduced, gammas):
        target = target + g * sc
        b_i = zrs.rs_eq_ind(sumcheck.build_eq(x_full[1:]), g * eq_r_dprime)  # γ baked in
        b_combined = b_i if b_combined is None else b_combined + b_i

    ok_lig = zorch_ligerito.verify_flock_ligerito(
        cfg, root, b_combined, target, pcs_open.ligerito_obj, ch)
    return ok & ok_lig


def verify(cfg, root, statement, res, a0, b0, m, k_log, k_skip, ch):
    """Verify a `prove_fast` result. Returns the scalar `ok`."""
    lc_proof = LincheckProof(rounds=res.lincheck[0], z_partial=res.lincheck[1])
    ab, c, ok_core = verify_core(cfg, root, statement, res.zerocheck, lc_proof,
                                 a0, b0, m, k_log, k_skip, ch)
    ok_open = verify_claims_ligerito(cfg, root, [ab, c], res.pcs_open, ch)
    return ok_core & ok_open
