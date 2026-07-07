"""FRI helpers shared by the BaseFold and Ligerito PCS opens — the row-batch
collapse, the epoch-arity schedule, and the query-count defaults. A leaf module
(depends only on `field`), so both the open frontend (`pcs_open`) and the
backend (`basefold`) import DOWN from it.

The per-round codeword fold itself is zorch's `coding.AdditiveReedSolomon.fold`
(see `basefold.py`); the deferred row-batch below stays flock-side because it
collapses `2^log_batch_size` interleaved lanes per position, a step outside the
FoldableCode seam. Requires `jax_enable_x64`.
"""
from __future__ import annotations

from flock_zorch import field

LOG_FRI_ARITY = 6        # flock pcs/commit.rs:26
DEFAULT_FRI_QUERIES = 243


def compute_fri_arities(log_dim: int) -> list[int]:
    """Decompose log_dim FRI rounds into epoch arities, each ≤ LOG_FRI_ARITY
    (flock `compute_fri_arities`): e.g. 17→[6,6,5], 12→[6,6], 13→[6,6,1]."""
    arities, rem = [], log_dim
    while rem > 0:
        a = min(rem, LOG_FRI_ARITY)
        arities.append(a)
        rem -= a
    return arities


def default_fri_queries(log_inv_rate: int) -> int:
    """flock `default_fri_queries`: rate 1/2 → 243, rate 1/4 → 148."""
    return {1: DEFAULT_FRI_QUERIES, 2: 148}[log_inv_rate]


def row_batch_fold_all(codeword, challenges, mul=field.mul):
    """Collapse each codeword position's `2^len(challenges)` lanes to one F128 via
    nested folds `buf[j] = u + r·(u+v)` (flock `row_batch_fold_all`).

    codeword: uint64 [n_pos·num_ntts, 2] (SoA, position-major); challenges:
    uint64 [log_batch_size, 2]. Returns uint64 [n_pos, 2]."""
    lbs = int(challenges.shape[0])
    num_ntts = 1 << lbs
    n_pos = codeword.shape[0] // num_ntts
    buf = codeword.reshape(n_pos, num_ntts, 2)
    length = num_ntts
    for i in range(lbs):
        r = challenges[i]
        half = length // 2
        br = buf.reshape(n_pos, half, 2, 2)
        u, v = br[:, :, 0, :], br[:, :, 1, :]
        buf = field.add(u, mul(r, field.add(u, v)))       # [n_pos, half, 2]
        length = half
    return buf.reshape(n_pos, 2)
