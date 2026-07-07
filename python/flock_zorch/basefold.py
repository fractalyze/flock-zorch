"""flock's BaseFold PCS open prover (`pcs::basefold::prove`), authored in jax —
byte-identical to flock-core. The keystone of the PCS open: a sumcheck over
(a_init, b) interleaved with codeword folding (deferred row-batch over the first
`log_batch_size` rounds, then per-round FRI folds), per-epoch Merkle commits, and
query openings.

Reuses flock-zorch's verified primitives: `sumcheck.fold_single` (the low-bit
a/b fold), `fri.{row_batch_fold_all, compute_fri_arities}`, and
`merkle.{merkle_tree, merkle_multi_proof}` + the SHA-256 `Challenger`. The
per-round codeword fold is zorch's `coding.AdditiveReedSolomon.fold` (its
LCH-basis twiddle schedule byte-matches flock's `ntt.compute_twiddles`; see
`coding_oracle_test`), run on the native `binary_field_ghash` dtype instead of
the SoA software/clmad field ops.

`target`/running_target do NOT affect the proof bytes (the proof carries the
round messages, commitments, final values, and query openings — not the running
target), so they're omitted. Root observes use `root_to_f128` = the FIRST 16
bytes of the 32-byte root. Requires `jax_enable_x64`.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from zorch.coding.additive_reed_solomon import AdditiveReedSolomon

from flock_zorch import field, sumcheck, merkle, fri

LABEL = b"flock-basefold-v0"


def _round_message(a, b, mul):
    """Round message (u_0, u_2) = (Σ a_e·b_e, Σ (a_e+a_o)·(b_e+b_o)) over the
    even/odd split (flock's round-0 prime / fused next-round message). Returns
    jnp (device) — the caller converts to np for the transcript/proof."""
    ae, ao = a[0::2], a[1::2]
    be, bo = b[0::2], b[1::2]
    u0 = field.sum(mul(ae, be))
    u2 = field.sum(mul(field.add(ae, ao), field.add(be, bo)))
    return u0, u2


# Per-round field ops jitted (cache per mul) so each round is ONE fused kernel,
# not eager op-by-op dispatch. The codeword fold lives on `code.fold` (below),
# on the ghash dtype, so it is not part of this mul-keyed bundle.
_BF_CACHE: dict = {}


def _bf_ops(mul):
    o = _BF_CACHE.get(mul)
    if o is None:
        o = (
            jax.jit(lambda a, b: _round_message(a, b, mul)),
            jax.jit(lambda a, r: sumcheck.fold_single(a, r, mul)),
            jax.jit(lambda cw, ch: fri.row_batch_fold_all(cw, ch, mul)),
        )
        _BF_CACHE[mul] = o
    return o


def _to_ghash(soa):
    """uint64 SoA (last axis size 2) -> binary_field_ghash on device."""
    return lax.bitcast_convert_type(jnp.asarray(soa), jnp.binary_field_ghash)


def _from_ghash(g):
    """binary_field_ghash -> uint64 SoA [n, 2] via host bytes (the device
    ghash->uint64 bitcast returns zeros, zorch#399)."""
    arr = np.asarray(g)
    return np.frombuffer(arr.tobytes(), np.uint64).reshape(arr.shape[0], 2)


def _leaf_bytes(codeword_np, n_leaves, leaf_f128):
    """codeword uint64 [.,2] -> uint8 [n_leaves, leaf_f128*16] (LE F128 bytes)."""
    return codeword_np.reshape(n_leaves, leaf_f128 * 2).view(np.uint8)


def _root_f128(root):
    """flock root_to_f128: F128 from the FIRST 16 bytes of the 32-byte root."""
    return root[:16].view(np.uint64)


def prove(z_packed, b, codeword, initial_tree, k_code, log_inv_rate, log_batch_size,
          n_queries, ch, mul=field.mul, use_host_sha: bool = False) -> dict:
    """Run BaseFold open on the SHARED challenger `ch` (so it composes in
    pcs::open after ring-switch). z_packed=a_init uint64 [2^log_msg,2]; b same;
    codeword uint64 [2^k_code · num_ntts, 2]; initial_tree uint8 [2·n_leaves-1, 32]
    (from commit). Returns the BaseFoldProof fields, byte-identical to flock."""
    log_dim = k_code - log_inv_rate
    log_msg = log_batch_size + log_dim
    num_ntts = 1 << log_batch_size
    arities = fri.compute_fri_arities(log_dim)
    num_epochs = len(arities)
    num_fri_commits = max(num_epochs - 1, 0)
    arity_0 = arities[0] if arities else 0
    post_rb_leaf_f128 = 1 << arity_0
    # The fold chain rides ONE additive-RS instance (block_len = 2^k_code); its
    # LCH twiddles are anchored to that block_len, so a folded layer is not a
    # smaller code's codeword — every round must go through this instance.
    code = AdditiveReedSolomon(1 << log_dim, 1 << log_inv_rate,
                               jnp.binary_field_ghash) if log_dim > 0 else None
    fold_fn = jax.jit(code.fold) if code is not None else None

    ch.observe_label(LABEL)
    round_message, fold_single, row_batch = _bf_ops(mul)

    a = jnp.asarray(z_packed)
    bb = jnp.asarray(b)
    cw_full = jnp.asarray(codeword)        # initial SoA codeword (kept for T1 leaves)
    cw_active = cw_full
    cw_g = None                            # ghash fold state; entered on the first FRI round
    round_messages, rb_challenges, round_commitments = [], [], []
    post_rb_codeword = post_rb_tree = None
    post_rb_root = np.zeros(32, np.uint8)
    epoch_codewords, epoch_trees, epoch_leaf_f128s = [], [], []
    current_epoch = rounds_in_epoch = 0

    # ---- interleaved (a,b) sumcheck + codeword fold + per-epoch Merkle commit ----
    # Each round: send (u0,u2), fold a/b at r; the first log_batch_size rounds defer
    # a row-batch over the codeword, the rest are per-round FRI folds, committed per epoch.
    for rnd in range(log_msg):
        u0, u2 = (np.asarray(v) for v in round_message(a, bb))
        ch.observe_f128(u0)
        ch.observe_f128(u2)
        round_messages.append((u0, u2))
        r = jnp.asarray(ch.sample_f128())
        a = fold_single(a, r)
        bb = fold_single(bb, r)

        if rnd < log_batch_size:
            rb_challenges.append(r)
            if rnd + 1 == log_batch_size:
                cw_active = row_batch(cw_full, jnp.stack(rb_challenges))
                if arities:
                    cw_np = np.asarray(cw_active)
                    n_leaves = cw_np.shape[0] // post_rb_leaf_f128
                    post_rb_tree = merkle.merkle_tree(_leaf_bytes(cw_np, n_leaves, post_rb_leaf_f128),
                                                      use_host_sha=use_host_sha)
                    post_rb_root = post_rb_tree[-1]
                    ch.observe_f128(_root_f128(post_rb_root))
                    post_rb_codeword = cw_np
        else:
            if cw_g is None:
                cw_g = _to_ghash(cw_active)      # enter the additive-RS fold domain
            cw_g = fold_fn(cw_g, _to_ghash(r.reshape(1, 2)))
            rounds_in_epoch += 1
            if rounds_in_epoch == arities[current_epoch]:
                if current_epoch + 1 < num_epochs:
                    leaf_f128 = 1 << arities[current_epoch + 1]
                    cw_np = _from_ghash(cw_g)
                    n_leaves = cw_np.shape[0] // leaf_f128
                    tree = merkle.merkle_tree(_leaf_bytes(cw_np, n_leaves, leaf_f128),
                                              use_host_sha=use_host_sha)
                    ch.observe_f128(_root_f128(tree[-1]))
                    round_commitments.append(tree[-1])
                    epoch_codewords.append(cw_np)
                    epoch_trees.append(tree)
                    epoch_leaf_f128s.append(leaf_f128)
                rounds_in_epoch = 0
                current_epoch += 1

    final_a = np.asarray(a)[0]
    final_b = np.asarray(bb)[0]
    final_codeword = _from_ghash(cw_g) if cw_g is not None else np.asarray(cw_active)

    # ---- query openings: sample positions, gather each layer's leaf, Merkle multi-proofs ----
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
