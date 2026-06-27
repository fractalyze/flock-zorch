"""flock's BaseFold PCS open prover (`pcs::basefold::prove`), authored in jax —
byte-identical to flock-core. The keystone of the PCS open: a sumcheck over
(a_init, b) interleaved with codeword folding (deferred row-batch over the first
`log_batch_size` rounds, then per-round FRI folds), per-epoch Merkle commits, and
query openings.

Reuses flock-zorch's verified primitives: `sumcheck.fold_single` (the low-bit
a/b fold), `pcs_open.{row_batch_fold_all, fri_fold, compute_fri_arities}`,
`merkle.{merkle_tree, merkle_multi_proof}`, `ntt.compute_twiddles`, and the
SHA-256 `Challenger`.

`target`/running_target do NOT affect the proof bytes (the proof carries the
round messages, commitments, final values, and query openings — not the running
target), so they're omitted. Root observes use `root_to_f128` = the FIRST 16
bytes of the 32-byte root. Requires `jax_enable_x64`.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from flock_zorch import field, sumcheck, merkle, ntt as ntt_mod, pcs_open

LABEL = b"flock-basefold-v0"


def _prime(a, b, mul):
    """Round message (u_0, u_2) = (Σ a_e·b_e, Σ (a_e+a_o)·(b_e+b_o)) over the
    even/odd split (flock's round-0 prime / fused next-round message)."""
    ae, ao = a[0::2], a[1::2]
    be, bo = b[0::2], b[1::2]
    u0 = sumcheck._xor_reduce(mul(ae, be))
    u2 = sumcheck._xor_reduce(mul(field.add(ae, ao), field.add(be, bo)))
    return np.asarray(u0), np.asarray(u2)


def _leaf_bytes(codeword_np, n_leaves, leaf_f128):
    """codeword uint64 [.,2] -> uint8 [n_leaves, leaf_f128*16] (LE F128 bytes)."""
    return codeword_np.reshape(n_leaves, leaf_f128 * 2).view(np.uint8)


def _root_f128(root):
    """flock root_to_f128: F128 from the FIRST 16 bytes of the 32-byte root."""
    return root[:16].view(np.uint64)


def prove(z_packed, b, codeword, initial_tree, k_code, log_inv_rate, log_batch_size,
          n_queries, ch, mul=field.mul) -> dict:
    """Run BaseFold open on the SHARED challenger `ch` (so it composes in
    pcs::open after ring-switch). z_packed=a_init uint64 [2^log_msg,2]; b same;
    codeword uint64 [2^k_code · num_ntts, 2]; initial_tree uint8 [2·n_leaves-1, 32]
    (from commit). Returns the BaseFoldProof fields, byte-identical to flock."""
    log_dim = k_code - log_inv_rate
    log_msg = log_batch_size + log_dim
    num_ntts = 1 << log_batch_size
    arities = pcs_open.compute_fri_arities(log_dim)
    num_epochs = len(arities)
    num_fri_commits = max(num_epochs - 1, 0)
    arity_0 = arities[0] if arities else 0
    post_rb_leaf_f128 = 1 << arity_0
    twiddles = jnp.asarray(ntt_mod.compute_twiddles(k_code)) if log_dim > 0 else None

    ch.observe_label(LABEL)

    a = jnp.asarray(z_packed)
    bb = jnp.asarray(b)
    cw_full = jnp.asarray(codeword)        # initial SoA codeword (kept for T1 leaves)
    cw_active = cw_full
    round_messages, rb_challenges, round_commitments = [], [], []
    post_rb_codeword = post_rb_tree = None
    post_rb_root = np.zeros(32, np.uint8)
    epoch_codewords, epoch_trees, epoch_leaf_f128s = [], [], []
    current_epoch = rounds_in_epoch = 0

    for rnd in range(log_msg):
        u0, u2 = _prime(a, bb, mul)
        ch.observe_f128(u0)
        ch.observe_f128(u2)
        round_messages.append((u0, u2))
        r = jnp.asarray(ch.sample_f128())
        a = sumcheck.fold_single(a, r, mul)
        bb = sumcheck.fold_single(bb, r, mul)

        if rnd < log_batch_size:
            rb_challenges.append(r)
            if rnd + 1 == log_batch_size:
                cw_active = pcs_open.row_batch_fold_all(cw_full, jnp.stack(rb_challenges), mul)
                if arities:
                    cw_np = np.asarray(cw_active)
                    n_leaves = cw_np.shape[0] // post_rb_leaf_f128
                    post_rb_tree = merkle.merkle_tree(_leaf_bytes(cw_np, n_leaves, post_rb_leaf_f128))
                    post_rb_root = post_rb_tree[-1]
                    ch.observe_f128(_root_f128(post_rb_root))
                    post_rb_codeword = cw_np
        else:
            fri_round_idx = rnd - log_batch_size
            layer = k_code - fri_round_idx - 1
            cw_active = pcs_open.fri_fold(cw_active, twiddles, layer, r, mul)
            rounds_in_epoch += 1
            if rounds_in_epoch == arities[current_epoch]:
                if current_epoch + 1 < num_epochs:
                    leaf_f128 = 1 << arities[current_epoch + 1]
                    cw_np = np.asarray(cw_active)
                    n_leaves = cw_np.shape[0] // leaf_f128
                    tree = merkle.merkle_tree(_leaf_bytes(cw_np, n_leaves, leaf_f128))
                    ch.observe_f128(_root_f128(tree[-1]))
                    round_commitments.append(tree[-1])
                    epoch_codewords.append(cw_np)
                    epoch_trees.append(tree)
                    epoch_leaf_f128s.append(leaf_f128)
                rounds_in_epoch = 0
                current_epoch += 1

    final_a = np.asarray(a)[0]
    final_b = np.asarray(bb)[0]
    final_codeword = np.asarray(cw_active)

    cw_full_np = np.asarray(cw_full)
    queries, init_pos, post_rb_pos = [], [], []
    epoch_pos = [[] for _ in range(num_fri_commits)]
    for _ in range(n_queries):
        raw = ch.sample_f128()
        position = int(raw[0]) & ((1 << k_code) - 1)
        init_leaf = cw_full_np[position * num_ntts:(position + 1) * num_ntts]
        init_pos.append(position)
        if arities:
            li = position >> arity_0
            post_rb_pos.append(li)
            post_rb_leaf = post_rb_codeword[li * post_rb_leaf_f128:(li + 1) * post_rb_leaf_f128]
        else:
            post_rb_leaf = np.zeros((0, 2), np.uint64)
        epoch_leaves, cum = [], arity_0
        for i in range(num_fri_commits):
            lf = epoch_leaf_f128s[i]
            li = (position >> cum) // lf
            epoch_leaves.append(epoch_codewords[i][li * lf:(li + 1) * lf])
            epoch_pos[i].append(li)
            cum += arities[i + 1]
        queries.append((position, init_leaf, post_rb_leaf, epoch_leaves))

    n_init_leaves = cw_full_np.shape[0] // num_ntts
    init_mp = merkle.merkle_multi_proof(initial_tree, n_init_leaves, init_pos)
    if arities:
        n_lv = post_rb_codeword.shape[0] // post_rb_leaf_f128
        post_rb_mp = merkle.merkle_multi_proof(post_rb_tree, n_lv, post_rb_pos)
    else:
        post_rb_mp = np.zeros((0, 32), np.uint8)
    epoch_mps = [merkle.merkle_multi_proof(epoch_trees[i], epoch_codewords[i].shape[0] // epoch_leaf_f128s[i],
                                           epoch_pos[i]) for i in range(num_fri_commits)]

    return {
        "round_messages": round_messages,
        "post_row_batch_commit": post_rb_root,
        "round_commitments": round_commitments,
        "final_a": final_a,
        "final_b": final_b,
        "final_codeword": final_codeword,
        "queries": queries,
        "initial_multi_proof": init_mp,
        "post_row_batch_multi_proof": post_rb_mp,
        "epoch_multi_proofs": epoch_mps,
    }
