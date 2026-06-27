"""flock additive NTT over F128 (LCH novel-polynomial basis), authored in jax.

Ports flock-core's `AdditiveNttF128::forward_transform_scalar` (the reference
oracle in `ntt/additive_ntt_f128.rs`). The forward transform maps novel-basis
polynomial coefficients to evaluations over an F2-affine subspace; it is `log_d`
sequential layers, each a fully data-parallel butterfly. Used by the PCS commit
and FRI folding.

Per flock's convention the per-(layer, block) twiddles are a small *sequential*
precompute (the subspace-polynomial recurrence); they are computed on the host
and passed in -- see `compute_twiddles`. This keeps the device kernel a pure
data-parallel butterfly.

F128 = uint64 [..., 2] (see field.py). Butterfly (LCH, neighbors-last):

    u' = u + v * t ;   v' = v + u'

At layer l there are 2^l blocks of size 2^(log_d - l); each block splits into two
halves of 2^(log_d - l - 1) paired across the half boundary with the block's
twiddle. Layer 0 pairs are N/2 apart; the deepest layer's are adjacent.

Requires `jax_enable_x64`.
"""
from __future__ import annotations

import functools

import numpy as np
import jax.numpy as jnp

from flock_zorch import field
from flock_zorch import _hostfield as hf


@functools.lru_cache(maxsize=None)
def compute_twiddles(log_d: int) -> np.ndarray:
    """Host-side twiddle table for the standard NTT, basis {1, x, ..., x^(log_d-1)}.

    Mirrors flock's `AdditiveNttF128::standard(log_d)` exactly: builds the
    subspace-polynomial eval table `W_i` via the recurrence
    `W_i(z) = W_{i-1}(z)*(W_{i-1}(z)+W_{i-1}(beta_{i-1}))`, normalizes each row by
    `W_i(beta_i)`, then emits `twiddle(layer, block) = span_get(evals[L-layer-1][1:],
    block)` layer-major. Returns uint64 [2^log_d - 1, 2]. Sequential (host).

    Memoized on `log_d` (the table is data-independent and the host recurrence is
    ~55ms at log_d=15) and returned read-only so the shared instance can't be
    mutated; every caller copies it to device via `jnp.asarray`.
    """
    L = log_d
    evals = [[1 << i for i in range(L)]]  # evals[0] = basis (x^i = bit i)
    for i in range(1, L):
        prev = evals[i - 1]
        evals.append([hf.mul(prev[k], prev[k] ^ prev[0]) for k in range(1, len(prev))])
    for row in evals:
        inv0 = hf.inv(row[0])
        for j in range(len(row)):
            row[j] = hf.mul(row[j], inv0)

    tw: list[int] = []
    for layer in range(log_d):
        basis_row = evals[L - layer - 1][1:]  # length == layer
        cur = [0]
        for bj in basis_row:  # subset-XOR doubling: cur[block] = span_get(basis_row, block)
            cur = cur + [c ^ bj for c in cur]
        tw.extend(cur)

    arr = np.empty((len(tw), 2), dtype=np.uint64)
    for idx, e in enumerate(tw):
        arr[idx, 0] = np.uint64(e & 0xFFFFFFFFFFFFFFFF)
        arr[idx, 1] = np.uint64((e >> 64) & 0xFFFFFFFFFFFFFFFF)
    arr.flags.writeable = False  # shared cached instance — callers jnp.asarray it
    return arr


def forward_transform_interleaved(data, twiddles, log_d: int, num_ntts: int, mul=field.mul):
    """Interleaved forward additive NTT — `num_ntts` independent size-2^log_d
    sub-NTTs sharing twiddles, the per-lane generalization of
    `forward_transform_scalar`. Ports flock's `forward_transform_interleaved`,
    used by `pcs::commit` (RS-encode every row-batch lane).

    data: uint64 [2^log_d * num_ntts, 2], SoA position-major
    (`data[pos*num_ntts + lane]`). twiddles: same layer-major table as the single
    transform (all lanes share them). Returns the same shape. The butterfly and
    twiddle schedule are identical to `forward_transform_scalar`; the lane axis is
    just carried as an inner batch.
    """
    n = 1 << log_d
    x = data.reshape(n, num_ntts, 2)                   # [pos, lane, F128]
    for layer in range(log_d):
        num_blocks = 1 << layer
        bsh = 1 << (log_d - layer - 1)
        off = num_blocks - 1
        t = twiddles[off:off + num_blocks]             # [num_blocks, 2]
        xr = x.reshape(num_blocks, 2, bsh, num_ntts, 2)
        u = xr[:, 0]                                    # [num_blocks, bsh, lane, 2]
        v = xr[:, 1]
        tb = t[:, None, None, :]                        # broadcast over bsh + lanes
        new_u = field.add(u, mul(v, tb))
        new_v = field.add(v, new_u)
        x = jnp.stack([new_u, new_v], axis=1).reshape(n, num_ntts, 2)
    return x.reshape(n * num_ntts, 2)


def forward_transform_scalar(data, twiddles, log_d: int, mul=field.mul):
    """Forward additive NTT, in the flock `forward_transform_scalar` convention.

    data: uint64 [2^log_d, 2]. twiddles: uint64 [2^log_d - 1, 2], laid out
    layer-major (layer l occupies indices [2^l - 1, 2^(l+1) - 1), block-ordered).
    log_d: static Python int. Returns uint64 [2^log_d, 2].

    `mul` is the GF(2^128) multiply (default the readable `field.mul`); pass
    `field_clmad.mul` for the ~255x clmad path on GPU (byte-identical).
    """
    n = 1 << log_d
    x = data
    for layer in range(log_d):
        num_blocks = 1 << layer
        bsh = 1 << (log_d - layer - 1)
        off = num_blocks - 1  # 2^layer - 1
        t = twiddles[off:off + num_blocks]          # [num_blocks, 2]
        xr = x.reshape(num_blocks, 2, bsh, 2)
        u = xr[:, 0]                                 # [num_blocks, bsh, 2]
        v = xr[:, 1]
        tb = t[:, None, :]                           # broadcast over the half
        new_u = field.add(u, mul(v, tb))
        new_v = field.add(v, new_u)
        x = jnp.stack([new_u, new_v], axis=1).reshape(n, 2)
    return x
