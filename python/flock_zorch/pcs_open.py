"""PCS opening (BaseFold/Ligerito FRI) — the FRI codeword fold, authored in jax,
byte-identical to flock-core `pcs::basefold::fri_fold_codeword`.

The FRI fold is the dominant compute of the PCS opening (log_dim rounds, each
halving the codeword): a data-parallel butterfly over the codeword pairs, so it
inherits clmad on GPU like the forward NTT. `fold_pair(t,u,v,r)`:

    v' = v + u ;  u' = u + v'·t ;  out = u' + r·(u' + v')

with the per-block twiddle `t = twiddle(layer, i)` from the same additive-NTT
subspace recurrence as `ntt.compute_twiddles` (`twiddle(layer, i) =
twiddles[2^layer - 1 + i]`). Requires `jax_enable_x64`.
"""
from __future__ import annotations

import jax.numpy as jnp

from flock_zorch import field


def fri_fold(codeword, twiddles, layer: int, challenge, mul=field.mul):
    """One FRI fold: codeword uint64 [2^(layer+1), 2] -> [2^layer, 2].

    `twiddles` is the full layer-major table from `ntt.compute_twiddles(k_code)`;
    this fold uses `twiddle(layer, i) = twiddles[2^layer - 1 + i]`. Byte-identical
    to flock `fri_fold_codeword(codeword, ntt, layer, challenge)`.
    """
    new_len = codeword.shape[0] // 2
    off = (1 << layer) - 1
    tw = twiddles[off:off + new_len]              # twiddle(layer, i), i in [0,new_len)
    cw = codeword.reshape(new_len, 2, 2)          # pairs (2i, 2i+1)
    u_in, v_in = cw[:, 0, :], cw[:, 1, :]
    v = field.add(v_in, u_in)
    u = field.add(u_in, mul(v, tw))
    return field.add(u, mul(challenge, field.add(u, v)))
