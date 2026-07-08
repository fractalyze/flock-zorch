"""flock's BaseFold PCS open prover (`pcs::basefold::prove`), authored in jax —
byte-identical to flock-core. The keystone of the PCS open: a sumcheck over
(a_init, b) interleaved with codeword folding (deferred row-batch over the first
`log_batch_size` rounds, then per-round FRI folds), per-epoch Merkle commits, and
query openings.

Reuses flock-zorch's verified primitives: `sumcheck.fold_single` (the low-bit
a/b fold), `fri.{row_batch_fold_all, compute_fri_arities}`, and
`merkle.{merkle_tree, merkle_multi_proof}` + the SHA-256 `Challenger`. The
per-round codeword fold is zorch's `coding.AdditiveReedSolomon.fold` (its
LCH-basis twiddle schedule + fold byte-match flock's additive NTT / FRI goldens;
see `coding_oracle_test`), run on the native `binary_field_ghash` dtype instead
of the SoA software/clmad field ops.

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

from flock_zorch import field, sumcheck
from flock_zorch.hash import merkle
from flock_zorch.pcs import fri
from flock_zorch.field import _hostfield as hf

LABEL = b"flock-basefold-v0"


def _hf_mul(a, b):
    """Host GF(2^128) multiply on a uint64[2] (lo, hi) scalar pair — for the
    verify sumcheck replay, which runs on host numpy scalars (not device)."""
    return field._to_lohi(hf.mul(field._to_int(a), field._to_int(b)))


def _round_message(a, b):
    """Round message (u_0, u_2) = (Σ a_e·b_e, Σ (a_e+a_o)·(b_e+b_o)) over the
    even/odd split (flock's round-0 prime / fused next-round message). Returns
    jnp (device) — the caller converts to np for the transcript/proof."""
    ag, bg = field.to_ghash(a), field.to_ghash(b)
    ae, ao = ag[0::2], ag[1::2]
    be, bo = bg[0::2], bg[1::2]
    u0 = jnp.sum(ae * be)
    u2 = jnp.sum((ae + ao) * (be + bo))
    return field.from_ghash(u0), field.from_ghash(u2)


# Per-round field ops jitted (memoized) so each round is ONE fused kernel,
# not eager op-by-op dispatch. The codeword fold lives on `code.fold` (below),
# on the ghash dtype, so it is not part of this bundle.
_BF_OPS = None


def _bf_ops():
    global _BF_OPS
    if _BF_OPS is None:
        _BF_OPS = (
            jax.jit(lambda a, b: _round_message(a, b)),
            jax.jit(lambda a, r: sumcheck.fold_single(a, r)),
            jax.jit(lambda cw, ch: fri.row_batch_fold_all(cw, ch)),
        )
    return _BF_OPS


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
          n_queries, ch) -> dict:
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
    round_message, fold_single, row_batch = _bf_ops()

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
                    post_rb_tree = merkle.merkle_tree(_leaf_bytes(cw_np, n_leaves, post_rb_leaf_f128))
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


# ---------------------------------------------------------------------------
# Verifier (byte-anchored to flock-core `pcs::basefold::verify`)
# ---------------------------------------------------------------------------


def _leaf_hash_bytes(leaf_soa: np.ndarray) -> np.ndarray:
    """One query's leaf (uint64 SoA [leaf_f128, 2]) -> LE F128 bytes [leaf_f128*16]."""
    return np.ascontiguousarray(leaf_soa, np.uint64).view(np.uint8).reshape(-1)


def _fold_coset(code, buf_g, betas, input_layer, coset_idx, k_code):
    """Fold a batch of `2^a` cosets (one per query) down to one value each, via
    `a` successive zorch `code.fold_values` levels — flock's `fri_fold_coset`
    assembled from the FoldableCode seam (per-epoch local coset refold).

    buf_g: ghash [Q, 2^a]; betas: length-`a` ghash scalars; coset_idx: int [Q]
    (each coset's index in `input_layer`); returns ghash [Q]. The LCH twiddle at
    fold step k is `code._twiddles[(block>>(level+1)) - 1 + pos]` with
    `level = k_code - input_layer + k`, `pos = coset_idx*half + j` — the same
    schedule `code.fold` byte-matched to flock's FRI in #40."""
    ci = jnp.asarray(coset_idx, jnp.int64)
    for k, beta in enumerate(betas):
        q, width = buf_g.shape
        half = width // 2
        level = k_code - input_layer + k
        pairs = buf_g.reshape(q, half, 2)
        lo, hi = pairs[:, :, 0], pairs[:, :, 1]
        pos = (ci[:, None] * half + jnp.arange(half)[None, :]).reshape(-1)
        folded = code.fold_values(lo.reshape(-1), hi.reshape(-1), beta, pos, level)
        buf_g = folded.reshape(q, half)
    return buf_g[:, 0]


def verify(target, proof, initial_codeword_root, k_code, log_inv_rate,
           log_batch_size, ch) -> tuple[bool, dict]:
    """flock's BaseFold verifier (`pcs::basefold::verify`), authored in jax. Replays
    the sumcheck + multi-arity FRI consistency on the shared challenger `ch`, then
    batch-verifies the three Merkle categories through zorch's
    `pcs.fold.verify_openings` (fed per-query `Opening`s expanded from flock's
    octopus multi-proof, `merkle.multi_proof_to_paths`).

    `target` = the claimed sum Σ_e a[e]·b[e] (proof-external, like flock). Returns
    `(ok, info)` where `info["challenges"]` is the sumcheck challenge sequence and
    `info["reason"]` names the first failing check (or ""). No exceptions on a
    malformed proof — it rejects with a reason, mirroring flock's `VerifyError`."""
    def reject(reason, challenges=None):
        return False, {"reason": reason, "challenges": challenges}

    log_msg = len(proof["round_messages"])
    if log_batch_size > log_msg:
        return reject("InvalidProofShape:log_batch_size")
    log_dim = log_msg - log_batch_size
    if k_code != log_dim + log_inv_rate:
        return reject("InvalidProofShape:k_code")
    num_ntts = 1 << log_batch_size
    arities = fri.compute_fri_arities(log_dim)
    num_epochs = len(arities)
    num_fri_commits = max(num_epochs - 1, 0)

    ch.observe_label(LABEL)

    if len(proof["round_commitments"]) != num_fri_commits:
        return reject("InvalidProofShape:round_commitments")
    # #queries is a soundness parameter, not a prover choice (flock SECURITY note).
    if len(proof["queries"]) != fri.default_fri_queries(log_inv_rate):
        return reject("InvalidProofShape:n_queries")

    # ---- replay sumcheck + observe commitments in lockstep with the prover ----
    running_target = np.asarray(target, np.uint64)
    challenges = []
    current_epoch = rounds_in_epoch = 0
    for rnd in range(log_msg):
        u0, u2 = proof["round_messages"][rnd]
        u0 = np.asarray(u0, np.uint64); u2 = np.asarray(u2, np.uint64)
        ch.observe_f128(u0)
        ch.observe_f128(u2)
        r = np.asarray(ch.sample_f128(), np.uint64)
        challenges.append(r)
        # running_target = u0 + r·(running_target + u2) + r²·u2   (flock).
        u1 = running_target ^ u2
        rr = _hf_mul(r, r)
        running_target = (u0 ^ _hf_mul(r, u1)) ^ _hf_mul(rr, u2)

        if rnd + 1 == log_batch_size and arities:
            ch.observe_f128(_root_f128(np.asarray(proof["post_row_batch_commit"])))
        if rnd >= log_batch_size:
            rounds_in_epoch += 1
            if rounds_in_epoch == arities[current_epoch]:
                if current_epoch + 1 < num_epochs:
                    ch.observe_f128(_root_f128(np.asarray(proof["round_commitments"][current_epoch])))
                rounds_in_epoch = 0
                current_epoch += 1

    info = {"reason": "", "challenges": challenges}

    # ---- final sumcheck + codeword-constancy checks ----
    final_a = np.asarray(proof["final_a"], np.uint64)
    final_b = np.asarray(proof["final_b"], np.uint64)
    if not np.array_equal(_hf_mul(final_a, final_b), running_target):
        return reject("SumcheckFinalMismatch", challenges)
    final_cw = np.asarray(proof["final_codeword"], np.uint64)
    if final_cw.shape[0] != (1 << log_inv_rate):
        return reject("FinalCodewordNotConstant:len", challenges)
    if not np.all([np.array_equal(final_cw[i], final_cw[0]) for i in range(final_cw.shape[0])]):
        return reject("FinalCodewordNotConstant", challenges)
    if not np.array_equal(final_cw[0], final_a):
        return reject("SumcheckFriMismatch", challenges)

    # ---- resample query positions (challenger state matches prover) ----
    n_q = len(proof["queries"])
    positions = np.array([int(np.asarray(ch.sample_f128())[0]) & ((1 << k_code) - 1)
                          for _ in range(n_q)], dtype=np.int64)

    arity_0 = arities[0] if arities else 0
    ch_g = [_to_ghash(r.reshape(1, 2)).reshape(()) for r in challenges]  # ghash betas

    # Gather per-query leaves (uint64 SoA) once.
    q_pos = np.array([q[0] for q in proof["queries"]], dtype=np.int64)
    if not np.array_equal(q_pos, positions):
        return reject("FoldMismatch:position", challenges)
    init_leaves = np.stack([np.asarray(q[1], np.uint64) for q in proof["queries"]])  # [Q, num_ntts, 2]
    if init_leaves.shape[1] != num_ntts:
        return reject("InitialMerkleFailed:leaf_len", challenges)

    # Row-batch fold each query's T1 lanes -> one post-row-batch value (SoA field).
    if log_batch_size > 0:
        rb_ch = jnp.stack([jnp.asarray(challenges[i]) for i in range(log_batch_size)])
        prbv = np.asarray(fri.row_batch_fold_all(
            jnp.asarray(init_leaves.reshape(n_q * num_ntts, 2)), rb_ch))  # [Q, 2]
    else:
        prbv = init_leaves.reshape(n_q, 2)

    # ---- fold-consistency chain (compare in uint64 SoA to dodge ghash-== gap) ----
    if not arities:
        # log_dim == 0: no FRI. The post-row-batch value is the fold output; it
        # must sit at the queried position of the (constant) final codeword.
        if not np.array_equal(final_cw[positions], prbv):
            return reject("FoldMismatch:final", challenges)
    else:
        # ONE additive-RS instance drives every fold layer (its LCH twiddles are
        # anchored to block_len = 2^k_code), mirroring the prover.
        code = AdditiveReedSolomon(1 << log_dim, 1 << log_inv_rate, jnp.binary_field_ghash)
        post_rb_leaves = np.stack([np.asarray(q[2], np.uint64) for q in proof["queries"]])  # [Q, 2^a0, 2]
        inner = (positions & ((1 << arity_0) - 1))
        got = post_rb_leaves[np.arange(n_q), inner]  # [Q, 2]
        if not np.array_equal(got, prbv):
            return reject("FoldMismatch:t2_crosscheck", challenges)
        coset_g = _to_ghash(post_rb_leaves.reshape(-1, 2)).reshape(n_q, 1 << arity_0)
        expected_g = _fold_coset(code, coset_g,
                                 ch_g[log_batch_size:log_batch_size + arity_0],
                                 k_code, positions >> arity_0, k_code)
        expected = _from_ghash(expected_g)

        cum = arity_0
        for i in range(num_fri_commits):
            next_arity = arities[i + 1]
            leaves_i = np.stack([np.asarray(q[3][i], np.uint64) for q in proof["queries"]])  # [Q, 2^na, 2]
            if leaves_i.shape[1] != (1 << next_arity):
                return reject("InvalidProofShape:epoch_leaf_len", challenges)
            p_at = positions >> cum
            offset = p_at & ((1 << next_arity) - 1)
            got = leaves_i[np.arange(n_q), offset]  # [Q, 2]
            if not np.array_equal(got, expected):
                return reject(f"FoldMismatch:epoch{i}", challenges)
            leaf_g = _to_ghash(leaves_i.reshape(-1, 2)).reshape(n_q, 1 << next_arity)
            expected_g = _fold_coset(code, leaf_g,
                                     ch_g[log_batch_size + cum:log_batch_size + cum + next_arity],
                                     k_code - cum, p_at >> next_arity, k_code)
            expected = _from_ghash(expected_g)
            cum += next_arity

        p_final = positions >> cum
        if not np.array_equal(final_cw[p_final], expected):
            return reject("FoldMismatch:final", challenges)

    # ---- Merkle: expand octopus -> per-query Openings -> zorch verify_openings ----
    from zorch.commit.merkle import Opening
    legs = []

    def _leg(mp, num_leaves, leaf_positions, leaf_soa_stack, root):
        leaf_bytes = np.stack([_leaf_hash_bytes(leaf_soa_stack[j]) for j in range(n_q)])
        paths = merkle.multi_proof_to_paths(mp, num_leaves, list(leaf_positions), leaf_bytes)
        depth = paths.shape[1]
        opening = Opening(row=jnp.asarray(leaf_bytes),
                          path=[jnp.asarray(paths[:, k, :]) for k in range(depth)])
        return (jnp.asarray(np.asarray(root, np.uint8)), jnp.asarray(leaf_positions), opening)

    legs.append(_leg(np.asarray(proof["initial_multi_proof"]), 1 << k_code,
                     positions, init_leaves, initial_codeword_root))
    if arities:
        legs.append(_leg(np.asarray(proof["post_row_batch_multi_proof"]),
                         1 << (k_code - arity_0), positions >> arity_0,
                         post_rb_leaves, proof["post_row_batch_commit"]))
        cum = arity_0
        for i in range(num_fri_commits):
            next_arity = arities[i + 1]
            leaves_i = np.stack([np.asarray(q[3][i], np.uint64) for q in proof["queries"]])
            leaf_idx = (positions >> cum) >> next_arity
            legs.append(_leg(np.asarray(proof["epoch_multi_proofs"][i]),
                             1 << (k_code - cum - next_arity), leaf_idx,
                             leaves_i, proof["round_commitments"][i]))
            cum += next_arity

    if not merkle.verify_openings_flock(legs):
        return reject("MerkleFailed", challenges)

    return True, info
