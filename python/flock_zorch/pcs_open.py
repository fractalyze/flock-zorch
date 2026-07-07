"""Full single-claim PCS opening, authored in jax — byte-identical to flock-core
`pcs::open`: observe `flock-pcs-open-v0` → ring-switch (s_hat_v + rs_eq_ind=b +
target) → BaseFold. The entry point is `open()`; the FRI helpers it drives
(`row_batch_fold_all` / `compute_fri_arities` / `default_fri_queries`) live in
`fri.py` — a leaf module that this open frontend and the basefold backend both
import down from. The per-round codeword fold is zorch's
`coding.AdditiveReedSolomon.fold` (see `basefold.py`). Requires `jax_enable_x64`.
"""
from __future__ import annotations

from flock_zorch import field
from flock_zorch.fri import default_fri_queries


def open(z_packed, codeword, initial_tree, x_outer, k_code, log_inv_rate, log_batch_size,
         ch, mul=field.mul, use_host_sha: bool = False) -> dict:
    """Full single-claim PCS open, byte-identical to flock `pcs::open`: observe
    `flock-pcs-open-v0` → ring-switch (s_hat_v + rs_eq_ind=b + target) → BaseFold.

    Returns {ring_switch: s_hat_v, basefold: <BaseFoldProof fields>}. `ch` is the
    shared challenger (already carrying commit/zerocheck/lincheck state in e2e)."""
    from flock_zorch import ring_switch, basefold  # local import: avoid import cycle
    ch.observe_label(b"flock-pcs-open-v0")
    s_hat_v, rs_eq_ind, _target = ring_switch.prove(z_packed, x_outer, ch, mul=mul)  # target unused: not in the proof bytes
    n_queries = default_fri_queries(log_inv_rate)
    bf = basefold.prove(z_packed, rs_eq_ind, codeword, initial_tree, k_code,
                        log_inv_rate, log_batch_size, n_queries, ch, mul=mul,
                        use_host_sha=use_host_sha)
    return {"ring_switch": s_hat_v, "basefold": bf}
