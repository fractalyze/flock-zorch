"""Device CSC (column-sparse) fold for `lincheck.CscCircuit` — the perf machinery
kept out of lincheck.py so its protocol reads top-to-bottom. The transposed binary
matvec out[c] = XOR_{r:M[r,c]=1} eq[r] is a column-segmented XOR-reduce: sort the
flat nonzeros by column ONCE (host), then per fold run a device prefix-XOR scan +
segment diff + clean scatter-set — no atomics, so the skewed const_pin column is
not a hotspot. Byte-identical to a host scatter.

Requires jax_enable_x64.
"""
from __future__ import annotations

import functools

import numpy as np
import jax
import jax.numpy as jnp

U64 = jnp.uint64


def _flatten_nz(rows):
    """Row-major sparse {0,1} matrix (rows[r] = cols with a 1 in row r) -> flat
    nonzero (col, row) index arrays, for a transposed XOR-gather fold."""
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    cols = np.concatenate([np.asarray(r, np.int64) for r in rows])
    rowi = np.concatenate([np.full(len(r), i, np.int64) for i, r in enumerate(rows)])
    return cols, rowi


def _csc_segments(col, row):
    """Precompute the device segment-XOR-reduce plan for one sparse binary matrix
    M (flat nonzeros: M[row[i], col[i]] = 1). The transposed fold out[c] =
    Σ_{i:col[i]=c} eq[row[i]] is a segment-XOR-reduce keyed by column. Sort the
    nonzeros by column (host, ONCE) so each column is a contiguous run; record the
    gather order, each run's LAST index, and the distinct present columns. The
    per-fold device path (`_seg_xor_fold`) then needs only a gather + prefix scan +
    a clean scatter — no atomics (so the skewed const_pin column is not a hotspot).
    Returns device int32 arrays (row_sorted, seg_end, present) or None if empty."""
    if len(col) == 0:
        return None
    order = np.argsort(col, kind="stable")
    col_s = col[order]
    row_s = row[order].astype(np.int32)
    change = np.empty(len(col_s), dtype=bool)
    change[-1] = True
    change[:-1] = col_s[1:] != col_s[:-1]            # run boundaries (last-of-run)
    seg_end = np.nonzero(change)[0].astype(np.int32)
    present = col_s[seg_end].astype(np.int32)
    return jnp.asarray(row_s), jnp.asarray(seg_end), jnp.asarray(present)


@functools.partial(jax.jit, static_argnums=(4,))
def _seg_xor_fold(eq, row_sorted, seg_end, present, k):
    """Device transposed binary matvec out[c] = XOR_{i:col[i]=c} eq[row[i]], via a
    sorted prefix-XOR scan. Inclusive prefix-XOR P over the column-sorted gathered
    values; each column's reduce = P[seg_end] XOR P[prev seg_end] (XOR is its own
    inverse), scattered (set, no duplicates) into the dense [k,2] output."""
    vals = eq[row_sorted]                                          # [nnz, 2]
    pref = jax.lax.associative_scan(jnp.bitwise_xor, vals, axis=0)  # inclusive prefix XOR
    ends = pref[seg_end]                                           # cumulative through each run end
    prev = jnp.concatenate([jnp.zeros((1, 2), U64), ends[:-1]], axis=0)
    seg = jnp.bitwise_xor(ends, prev)                             # per-column XOR-reduce
    return jnp.zeros((k, 2), U64).at[present].set(seg)
