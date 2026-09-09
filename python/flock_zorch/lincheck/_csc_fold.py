"""Device CSC (column-sparse) fold for `lincheck.CscCircuit` — the perf machinery
kept out of lincheck/prover.py so its protocol reads top-to-bottom. The
transposed binary matvec out[c] = XOR_{r:M[r,c]=1} eq[r] is a column-segmented
XOR-reduce, and the two platforms reach it differently:

- GPU (`_csc_segments` + `_seg_xor_fold`): sort the flat nonzeros by column ONCE
  (host), then per fold run a device prefix-XOR scan + segment diff + clean
  scatter-set — no atomics, so the skewed const_pin column is not a hotspot.
- CPU (`_flat_nz` + `_scatter_xor_fold`): scatter-XOR the nonzeros straight into
  one accumulator, in the row-major order they were flattened in. XLA:CPU emits
  the scatter as a serial loop, so the scan's log-depth passes are pure overhead
  there, and the sort is not worth its locality (see `_flat_nz`).

Both are byte-identical to a host scatter, and to each other.

Requires jax_enable_x64.
"""

from __future__ import annotations

import functools

import frx
import frx.numpy as fnp
import numpy as np

from flock_zorch import ghash


def _flatten_nz(rows):
    """Row-major sparse {0,1} matrix (rows[r] = cols with a 1 in row r) -> flat
    nonzero (col, row) index arrays, for a transposed XOR-gather fold."""
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    cols = np.concatenate([np.asarray(r, np.int64) for r in rows])
    rowi = np.concatenate([np.full(len(r), i, np.int64) for i, r in enumerate(rows)])
    return cols, rowi


def _csc_segments(col, row):
    """Precompute the GPU segment-XOR-reduce plan for one sparse binary matrix M
    (flat nonzeros: M[row[i], col[i]] = 1); `_flat_nz` is the CPU counterpart.

    The transposed fold out[c] = Σ_{i:col[i]=c} eq[row[i]] is a segment-XOR-reduce
    keyed by column. Sort the nonzeros by column (host, ONCE) so each column is a
    contiguous run; record the gather order, each run's LAST index, and the
    distinct present columns. The per-fold device path (`_seg_xor_fold`) then
    needs only a gather + prefix scan + a clean scatter — no atomics, so the
    skewed const_pin column is not a hotspot.

    Returns device int32 arrays (row_sorted, seg_end, present) or None if empty."""
    if len(col) == 0:
        return None
    order = np.argsort(col, kind="stable")
    col_s = col[order]
    row_s = row[order].astype(np.int32)
    change = np.empty(len(col_s), dtype=bool)
    change[-1] = True
    change[:-1] = col_s[1:] != col_s[:-1]  # run boundaries (last-of-run)
    seg_end = np.nonzero(change)[0].astype(np.int32)
    present = col_s[seg_end].astype(np.int32)
    return fnp.asarray(row_s), fnp.asarray(seg_end), fnp.asarray(present)


@functools.partial(frx.jit, static_argnums=(4,))
def _seg_xor_fold(eq, row_sorted, seg_end, present, k):
    """Device transposed binary matvec out[c] = XOR_{i:col[i]=c} eq[row[i]], via a
    sorted prefix-XOR scan. Inclusive prefix-XOR P over the column-sorted gathered
    values; each column's reduce = P[seg_end] XOR P[prev seg_end] (XOR is its own
    inverse), scattered (set, no duplicates) into the dense [k,2] output."""
    vals = eq[row_sorted]  # ghash [nnz]
    pref = frx.lax.associative_scan(
        lambda a, b: a + b, vals, axis=0
    )  # inclusive prefix XOR (ghash add)
    ends = pref[seg_end]  # cumulative through each run end
    prev = fnp.concatenate([ghash.zeros(1), ends[:-1]], axis=0)
    seg = ends + prev  # per-column XOR-reduce (add is its own inverse)
    return ghash.zeros(k).at[present].set(seg)


def _flat_nz(col, row):
    """Precompute the CPU scatter-XOR plan for one sparse binary matrix M (flat
    nonzeros: M[row[i], col[i]] = 1). Unlike `_csc_segments` this keeps the
    nonzeros in the row-major order `_flatten_nz` produced — no sort, so there
    are no segments to diff and no scan to run; `_scatter_xor_fold` reads these
    two arrays directly.

    Leaving them row-major is the faster order, not merely the cheaper one.
    Column-sorting makes the accumulator writes sequential but the `eq` gather
    random; row-major does the reverse, and the reverse wins, because a row's
    whole run of nonzeros gathers the SAME eq[r] (row degrees reach 2,514 on the
    m26 blake3 A₀) while both the accumulator and eq are 256 KiB and L2-resident
    either way. Measured on that matrix, m26, 16 cores: 34.9 ms row-major vs
    50.6 ms column-sorted (and 51.1 ms with `indices_are_sorted`).

    Returns device int32 arrays (col, row) or None if empty."""
    if len(col) == 0:
        return None
    return fnp.asarray(col.astype(np.int32)), fnp.asarray(row.astype(np.int32))


@functools.partial(frx.jit, static_argnums=(3,))
def _scatter_xor_fold(eq, col, row, k):
    """Device transposed binary matvec out[c] = XOR_{i:col[i]=c} eq[row[i]], as
    one scatter-XOR into a single dense [k,2] accumulator (ghash add IS XOR, so
    the scatter-add is the XOR-reduce and duplicate columns accumulate).

    The CPU arm of `_seg_xor_fold`, and byte-identical to it. XLA:CPU lowers a
    scatter to a serial loop over the nonzeros, which is what makes this the
    cheaper shape there — the scan arm's log-depth passes buy parallelism the
    backend does not deliver — and equally what caps it: one core does the whole
    reduce (fractalyze/xla#679)."""
    return ghash.zeros(k).at[col].add(eq[row])
