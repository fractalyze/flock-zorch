"""flock's Ligerito recursive PCS, authored in jax — byte-identical to flock-core
`pcs::ligerito`. The PCS flock's headline hash benchmarks use: instead of a single
codeword folded over FRI rounds (BaseFold), it RE-COMMITS the partially-folded
witness at R recursive levels (shrinking RS rate), driving one continuous
interleaved sumcheck (fold / introduce / glue) with per-level OOD binding, query
PoW, and a plaintext residual. Proof scales ~log²(log_n).

Reuses flock-zorch's byte-identical primitives: `ntt` (compute_twiddles +
forward_transform_interleaved), `merkle` (tree/multi_proof + host SHA-NI),
`sumcheck` (build_eq / fold / _prime), `challenger` (SHA-256 FS + grind_pow),
`ring_switch.prove_batched`. This module adds the recursive driver.

Port status (M0): `ligero_commit` (per-level RS-encode + Merkle). The SumcheckProver
driver, induce_sumcheck_poly + LCH basis, and the recursion loop land in M1..M7.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from flock_zorch import field, ntt as ntt_mod, merkle, sumcheck, basefold


def ceil_log2(x: int) -> int:
    return (x - 1).bit_length()


def sample_distinct_queries(ch, block_len: int, count: int) -> list[int]:
    """flock `sample_distinct_queries`: sample F128, take (v.lo % block_len) until
    `count` distinct, then sort ascending. Same challenger draw order as flock."""
    assert count <= block_len
    seen, out = set(), []
    while len(out) < count:
        v = ch.sample_f128()
        q = int(v[0]) % block_len
        if q not in seen:
            seen.add(q); out.append(q)
    out.sort()
    return out


class SumcheckProver:
    """flock `pcs::ligerito::SumcheckProver` — the stateful interleaved sumcheck
    driver over (f, combined_basis, t_r). Round message (u_0,u_2) == basefold._prime
    (LSB even/odd split); fold == sumcheck.fold_single (LSB). introduce_new stages a
    fresh basis; glue folds it into combined_basis with separation α."""

    def __init__(self, f, b1, h1, mul):
        self.mul = mul
        self.f = jnp.asarray(f)
        self.combined_basis = jnp.asarray(b1)
        self.t_r = np.asarray(h1, np.uint64).reshape(2)
        self.transcript: list = []
        self._pending = None

    def _msg(self, a, b):
        u0, u2 = basefold._prime(a, b, self.mul)
        m = (np.asarray(u0), np.asarray(u2))
        self.transcript.append(m)
        return m

    @classmethod
    def new(cls, f, b1, h1, mul):
        s = cls(f, b1, h1, mul)
        return s, s._msg(s.f, s.combined_basis)

    @classmethod
    def new_with_first_msg(cls, f, b1, h1, first_msg, mul):
        s = cls(f, b1, h1, mul)
        s.transcript.append(first_msg)
        return s, first_msg

    def fold(self, r):
        rj = jnp.asarray(r)
        self.f = sumcheck.fold_single(self.f, rj, self.mul)
        self.combined_basis = sumcheck.fold_single(self.combined_basis, rj, self.mul)
        return self._msg(self.f, self.combined_basis)

    def introduce_new(self, b_new, h_new):
        bn = jnp.asarray(b_new)
        msg = self._msg(self.f, bn)
        self._pending = (bn, np.asarray(h_new, np.uint64).reshape(2))
        return msg

    def introduce_new_with_eval(self, b_new):
        bn = jnp.asarray(b_new)
        h_new = np.asarray(sumcheck._xor_reduce(self.mul(self.f, bn), axis=0))  # Σ f·b_new
        msg = self._msg(self.f, bn)
        self._pending = (bn, h_new)
        return msg, h_new

    def glue(self, alpha):
        bn, h_new = self._pending
        self._pending = None
        a = jnp.asarray(alpha)
        self.combined_basis = field.add(self.combined_basis, self.mul(a, bn))
        self.t_r = np.asarray(field.add(jnp.asarray(self.t_r), self.mul(a, jnp.asarray(h_new))))


def ligero_commit(poly, log_msg_cols: int, log_num_interleaved: int, log_inv_rate: int,
                  mul=field.mul, use_host_sha: bool = False):
    """Per-level Ligero commit (flock `pcs::ligerito::ligero_commit`): reshape `poly`
    (len num_interleaved·msg_cols, SoA `poly[col·num_interleaved + lane]`) into a
    block_len × num_interleaved matrix, RS-encode each lane via the additive NTT
    (zero-pad msg_cols→block_len then forward-transform — byte-identical to flock's
    replicate-+-start-at-layer form, as the first log_inv_rate layers are copies),
    and Merkle-commit the rows (leaf = num_interleaved F128). Returns (mat, tree).

    mat: uint64 [block_len·num_interleaved, 2] (SoA); tree: uint8 [2·block_len-1, 32]."""
    msg_cols = 1 << log_msg_cols
    num_int = 1 << log_num_interleaved
    block_len = msg_cols << log_inv_rate
    log_block_len = log_msg_cols + log_inv_rate

    x = jnp.asarray(poly).reshape(msg_cols, num_int, 2)
    pad = jnp.zeros((block_len - msg_cols, num_int, 2), dtype=x.dtype)
    cw = jnp.concatenate([x, pad], axis=0).reshape(block_len * num_int, 2)
    tw = jnp.asarray(ntt_mod.compute_twiddles(log_block_len))
    cw = ntt_mod.forward_transform_interleaved(cw, tw, log_block_len, num_int, mul=mul)
    mat = np.asarray(cw)
    leaves = mat.reshape(block_len, num_int * 2).view(np.uint8)
    tree = merkle.merkle_tree(leaves, use_host_sha=use_host_sha)
    return mat, tree
