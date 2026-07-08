"""flock's BaseFold PCS open VERIFIER (`pcs::basefold::verify`), authored in jax
— byte-anchored to flock-core (the prover moved to `zorch_basefold.py`,
flock-zorch#50 task 8a; this module is now verify-only).

Reuses flock-zorch's verified primitives: `fri.{row_batch_fold_all,
compute_fri_arities}` and `merkle.{multi_proof_to_paths, verify_openings_flock}`
+ the SHA-256 `Challenger`. The per-round codeword fold is zorch's
`coding.AdditiveReedSolomon.fold` (its LCH-basis twiddle schedule + fold
byte-match flock's additive NTT / FRI goldens; see `coding_oracle_test`), run on
the native `binary_field_ghash` dtype instead of the SoA software/clmad field
ops.

`target`/running_target do NOT affect the proof bytes (the proof carries the
round messages, commitments, final values, and query openings — not the running
target), so they're omitted. Root observes use `root_to_f128` = the FIRST 16
bytes of the 32-byte root. Requires `jax_enable_x64`.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import lax

from zorch.coding.additive_reed_solomon import AdditiveReedSolomon

from flock_zorch import field, merkle, fri
from flock_zorch import _hostfield as hf

LABEL = b"flock-basefold-v0"


def _hf_mul(a, b):
    """Host GF(2^128) multiply on a uint64[2] (lo, hi) scalar pair — for the
    verify sumcheck replay, which runs on host numpy scalars (not device)."""
    return field._to_lohi(hf.mul(field._to_int(a), field._to_int(b)))


def _to_ghash(soa):
    """uint64 SoA (last axis size 2) -> binary_field_ghash on device."""
    return lax.bitcast_convert_type(jnp.asarray(soa), jnp.binary_field_ghash)


def _from_ghash(g):
    """binary_field_ghash -> uint64 SoA [n, 2] via host bytes (the device
    ghash->uint64 bitcast returns zeros, zorch#399)."""
    arr = np.asarray(g)
    return np.frombuffer(arr.tobytes(), np.uint64).reshape(arr.shape[0], 2)


def _root_f128(root):
    """flock root_to_f128: F128 from the FIRST 16 bytes of the 32-byte root."""
    return root[:16].view(np.uint64)


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
