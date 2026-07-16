"""flock's BaseFold PCS open (`pcs::basefold`), authored in frx — byte-identical
to flock-core. `prove` drives `zorch.pcs.basefold`'s injectable choreography seam
(config + sumcheck kernel + Fiat-Shamir choreography) over flock's shared
challenger, retiring the in-tree frx recursion; `verify` stays flock's own
byte-anchored verifier.

The open is a sumcheck over (a_init, b) interleaved with codeword folding
(deferred row-batch over the first `log_batch_size` rounds, then per-round FRI
folds), per-epoch Merkle commits, and query openings. flock's fold schedule is a
non-native cadence (`row_batch_prefix` + multi-arity `fold_arities`), so
`BasefoldProver.open_with_basis` returns a generic `CadenceProof` that
`_flock_basefold_proof` reshapes into flock's octopus `BasefoldProof` — the same
bytes the retired in-tree prover produced.

Three flock deltas ride the seam (one module per scheme, mirroring `pcs.ligerito`):
`FlockProductKernel` (the degree-2 `(a, b)` product round), `FlockBasefoldChoreography`
(flock's Fiat-Shamir shape — `flock-basefold-v0` label, `root_to_f128` observes, no
terminal observe, masked query positions), and `flock_basefold_config` (the schedule
as a `BasefoldConfig`).

`target`/running_target do NOT affect the proof bytes (the proof carries the round
messages, commitments, final values, and query openings — not the running target),
so they're omitted. Root observes use `root_to_f128` = the FIRST 16 bytes of the
32-byte root. Requires `jax_enable_x64`.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import frx
import frx.numpy as jnp
from frx import Array, lax

from zorch.coding.additive_reed_solomon import AdditiveReedSolomon
from zorch.pcs.basefold.choreography import BasefoldChoreography
from zorch.pcs.basefold.config import BasefoldConfig
from zorch.pcs.basefold.kernel import SumcheckKernel
from zorch.pcs.basefold.prover import BasefoldProver, BasefoldProverData

from flock_zorch import ghash, fs
from flock_zorch.hash import merkle
from flock_zorch.pcs import fri
from flock_zorch.pcs.ligerito import FlockTranscript


@dataclass(frozen=True)
class BasefoldProof:
    """flock's BaseFold opening proof: the per-round sumcheck messages, the FRI
    commitment roots (post-row-batch + per-epoch), the final folded a/b and
    codeword, and the per-leg Merkle multi-proofs over the `queries`."""

    round_messages: Any
    post_row_batch_commit: Any
    round_commitments: Any
    final_a: Any
    final_b: Any
    final_codeword: Any
    queries: Any
    initial_multi_proof: Any
    post_row_batch_multi_proof: Any
    epoch_multi_proofs: Any


LABEL = b"flock-basefold-v0"


# ---------------------------------------------------------------------------
# Prover — flock's deltas on zorch's BaseFold seam, driven through
# `BasefoldProver.open_with_basis` (byte-identical to the retired in-tree port).
# ---------------------------------------------------------------------------


def _fold_single(v: Array, r: Array) -> Array:
    """Bind the low variable of one multilinear at `r` on the ghash dtype:
    `out[x] = v[2x] + r·(v[2x] + v[2x+1])` (flock `fold_in_place_single`, the
    char-2 multilinear bind)."""
    pairs = v.reshape(-1, 2)
    v0, v1 = pairs[:, 0], pairs[:, 1]
    return v0 + r * (v0 + v1)


@dataclass(frozen=True)
class FlockProductKernel(SumcheckKernel):
    """flock's degree-2 product sumcheck round over the state `(a, b)`. The
    message is `(Σaₑbₑ, Σ(aₑ+aₒ)(bₑ+bₒ))` over the low-bit even/odd split; the
    fold binds the low variable of both vectors by the shared challenge; the
    terminal is `(a[0], b[0])`. No opening point, so `round_check` never runs
    (the flock verifier is untouched by the seam adoption)."""

    def initial_state(self, mle: Array, basis: Array, claim: Array) -> tuple:
        del claim  # flock's proof bytes don't carry the running target
        return (mle, basis)  # (a, b)

    def message(self, state: tuple) -> tuple[Array, Array]:
        a, b = state
        ae, ao = a[0::2], a[1::2]
        be, bo = b[0::2], b[1::2]
        u0 = jnp.sum(ae * be)
        u2 = jnp.sum((ae + ao) * (be + bo))
        return u0, u2

    def fold(self, state: tuple, message: tuple[Array, Array], r: Array) -> tuple:
        del message
        a, b = state
        return (_fold_single(a, r), _fold_single(b, r))

    def final(self, state: tuple) -> tuple[Array, Array]:
        a, b = state
        return (a[0], b[0])


# --- the flock BaseFold Fiat-Shamir choreography -----------------------------


@dataclass(frozen=True)
class FlockBasefoldChoreography(BasefoldChoreography):
    """flock `pcs::basefold::prove`'s FS shape over `FlockTranscript`: the
    `flock-basefold-v0` label bind, eager `(u0, u2)` emission, `root_to_f128`
    root observes, no terminal observe, and flock's masked query positions.
    No grind (flock's BaseFold grinds nothing)."""

    @property
    def eager_messages(self) -> bool:
        return True

    def bind_statement(
        self, transcript: FlockTranscript, root: Array, point, value: Array
    ) -> FlockTranscript:
        del root, point, value  # flock binds only the domain label here
        return FlockTranscript(transcript.inner.observe_label(LABEL))

    def fold_challenge(
        self, transcript: FlockTranscript, msg, level: int, fold_idx: int
    ) -> tuple[FlockTranscript, Array]:
        del msg, level, fold_idx  # eager: the (u0, u2) message is already absorbed
        transcript, r = transcript.sample(1)
        return transcript, r[0]

    def observe_root(self, transcript: FlockTranscript, root: Array) -> FlockTranscript:
        # root_to_f128: the F128 in the FIRST 16 bytes of the 32-byte root.
        f128 = lax.bitcast_convert_type(
            jnp.asarray(root)[:16], jnp.binary_field_ghash
        )
        return transcript.observe(f128)

    def observe_final(
        self, transcript: FlockTranscript, final_poly: Array
    ) -> FlockTranscript:
        # flock samples queries straight off the last fold challenge — the final
        # codeword is never observed.
        del final_poly
        return transcript

    def sample_queries(
        self, transcript: FlockTranscript, block_len: int, count: int
    ) -> tuple[FlockTranscript, Array]:
        # flock's BaseFold queries: `count` scalar F128 squeezes, each taken as
        # its low limb mod block_len (dups kept, order kept — NOT distinct, NOT
        # sorted). Uses the same `fs.sample_chain` primitive flock's verifier
        # re-samples positions with, so the two match by construction.
        inner, pos_g = fs.sample_chain(transcript.inner, count)
        positions = np.asarray(ghash.from_ghash_host(pos_g)[:, 0]) % block_len
        return FlockTranscript(inner), jnp.asarray(positions, jnp.int32)


def flock_basefold_config(
    k_code: int, log_inv_rate: int, log_batch_size: int
) -> BasefoldConfig:
    """flock's fold schedule as a `BasefoldConfig`: a `log_batch_size` row-batch
    prefix, then `compute_fri_arities(log_dim)` epochs. `num_vars` is the total
    sumcheck round count `log_batch_size + log_dim`."""
    log_dim = k_code - log_inv_rate
    return BasefoldConfig(
        num_vars=log_batch_size + log_dim,
        num_queries=fri.default_fri_queries(log_inv_rate),
        row_batch_prefix=log_batch_size,
        fold_arities=tuple(fri.compute_fri_arities(log_dim)),
    )


# --- wire assembly: zorch CadenceProof -> flock's BasefoldProof --------------


def _lohi(x: Array) -> np.ndarray:
    """ghash (any shape) -> (-1, 2) uint64 lo‖hi (the uint8 bitcast direction)."""
    b = np.asarray(lax.bitcast_convert_type(x, jnp.uint8)).tobytes()
    return np.frombuffer(b, np.uint64).reshape(-1, 2).copy()


def _f128_scalar(x: Array) -> np.ndarray:
    """device ghash scalar -> host `binary_field_ghash` scalar (same 16 wire bytes)."""
    b = np.asarray(lax.bitcast_convert_type(x, jnp.uint8)).tobytes()
    return np.frombuffer(b, ghash._GHASH_HOST).reshape(())


def _f128v(x: Array) -> np.ndarray:
    """device ghash (any shape) -> host `binary_field_ghash` [-1] (same wire bytes)."""
    b = np.asarray(lax.bitcast_convert_type(x, jnp.uint8)).tobytes()
    return np.frombuffer(b, ghash._GHASH_HOST).copy()


def _leaf_rows(row: Array) -> np.ndarray:
    """opened leaf rows ghash [Q, w] -> uint64 [Q, w, 2] (per-query F128 leaf)."""
    q, w = int(row.shape[0]), int(row.shape[1])
    return _lohi(row).reshape(q, w, 2)


def _octopus(opening, num_leaves: int, positions: Array) -> np.ndarray:
    """flock's octopus multi-proof from a zorch `Opening`'s per-query paths +
    the sampled positions (byte-identical to `merkle.merkle_multi_proof`)."""
    paths = np.stack([np.asarray(p, np.uint8) for p in opening.path], axis=1)
    return merkle.paths_to_multi_proof(paths, num_leaves, np.asarray(positions))


def _flock_basefold_proof(proof, k_code: int, log_inv_rate: int, num_ntts: int) -> BasefoldProof:
    """zorch `CadenceProof` -> flock's `BasefoldProof`, byte-identical to the
    in-tree `basefold.prove` return."""
    arities = fri.compute_fri_arities(k_code - log_inv_rate)
    num_fri_commits = max(len(arities) - 1, 0)
    has_prefix_commit = bool(arities)  # the post-row-batch commit exists iff FRI does

    round_messages = [
        (_f128_scalar(u0), _f128_scalar(u2)) for (u0, u2) in proof.round_messages
    ]
    roots = [np.asarray(r, np.uint8).copy() for r in proof.commit_roots]
    post_rb_root = roots[0] if has_prefix_commit else np.zeros(32, np.uint8)
    round_commitments = roots[1:]  # per-epoch roots (all but the last epoch)

    final_a = _f128_scalar(proof.final_state[0])
    final_b = _f128_scalar(proof.final_state[1])
    final_codeword = _f128v(proof.final_codeword)  # [2^log_inv_rate] ghash

    positions = np.asarray(proof.positions)
    n_q = int(positions.shape[0])

    # Layer order: 0 = initial, 1 = post-row-batch (if arities), 2+ = epochs.
    init_rows = _leaf_rows(proof.layer_openings[0].row)  # [Q, num_ntts, 2]
    if has_prefix_commit:
        post_rb_rows = _leaf_rows(proof.layer_openings[1].row)  # [Q, 2^arity0, 2]
        epoch_rows = [
            _leaf_rows(proof.layer_openings[2 + i].row) for i in range(num_fri_commits)
        ]
    else:
        post_rb_rows = np.zeros((n_q, 0, 2), np.uint64)
        epoch_rows = []

    queries = []
    for i in range(n_q):
        epoch_leaves = [epoch_rows[e][i] for e in range(num_fri_commits)]
        queries.append((int(positions[i]), init_rows[i], post_rb_rows[i], epoch_leaves))

    initial_mp = _octopus(
        proof.layer_openings[0], proof.layer_num_leaves[0], proof.layer_positions[0]
    )
    if has_prefix_commit:
        post_rb_mp = _octopus(
            proof.layer_openings[1], proof.layer_num_leaves[1], proof.layer_positions[1]
        )
        epoch_mps = [
            _octopus(
                proof.layer_openings[2 + i],
                proof.layer_num_leaves[2 + i],
                proof.layer_positions[2 + i],
            )
            for i in range(num_fri_commits)
        ]
    else:
        post_rb_mp = np.zeros((0, 32), np.uint8)
        epoch_mps = []

    return BasefoldProof(
        round_messages=round_messages,
        post_row_batch_commit=post_rb_root,
        round_commitments=round_commitments,
        final_a=final_a,
        final_b=final_b,
        final_codeword=final_codeword,
        queries=queries,
        initial_multi_proof=initial_mp,
        post_row_batch_multi_proof=post_rb_mp,
        epoch_multi_proofs=epoch_mps,
    )


def prove(
    z_packed,
    b_combined,
    codeword,
    initial_tree,
    k_code,
    log_inv_rate,
    log_batch_size,
    n_queries,
    ch,
) -> BasefoldProof:
    """Drive `zorch.pcs.basefold` over flock's shared challenger and assemble a
    flock `BasefoldProof` — byte-identical to the retired in-tree `basefold.prove`.

    a = z_packed, b = b_combined are the product sumcheck's two vectors; the
    codeword (the interleaved additive-NTT of z_packed) is folded down. The
    Fiat-Shamir rides flock's live `ch` (bridged into a `FlockTranscript` at its
    current state, written back after the open) so the BaseFold open continues
    the transcript ring-switch built. The initial matrix is re-committed inside
    the driver (byte-identical to flock's external commit); reusing the external
    `initial_tree` is a perf follow-up."""
    del initial_tree
    log_dim = k_code - log_inv_rate
    num_ntts = 1 << log_batch_size
    n_pos = 1 << k_code

    a = ghash.to_ghash(jnp.asarray(z_packed))  # [2^log_msg]
    b = b_combined  # native ghash [2^log_msg]
    cw_g = ghash.to_ghash(jnp.asarray(codeword))  # [n_pos * num_ntts]

    code = AdditiveReedSolomon(1 << log_dim, 1 << log_inv_rate, jnp.binary_field_ghash)
    config = replace(
        flock_basefold_config(k_code, log_inv_rate, log_batch_size),
        num_queries=int(n_queries),
    )
    prover = BasefoldProver(
        code,
        merkle.GHASH_TREE,
        num_queries=int(n_queries),
        choreography=FlockBasefoldChoreography(),
        kernel=FlockProductKernel(),
        config=config,
    )
    pd = BasefoldProverData(
        digest_layers=[],
        mle=a,
        codeword=cw_g,
        leaves=cw_g.reshape(n_pos, num_ntts),
        widths=(num_ntts,),
    )
    value = ghash.to_ghash(jnp.zeros(2, jnp.uint64))  # ignored (no target on the wire)

    proof, t = prover.open_with_basis(pd, b, value, FlockTranscript(ch._t))
    ch._t = t.inner
    return _flock_basefold_proof(proof, k_code, log_inv_rate, num_ntts)


# ---------------------------------------------------------------------------
# Verifier (byte-anchored to flock-core `pcs::basefold::verify`)
# ---------------------------------------------------------------------------


def _hf_mul(a, b):
    """Host GF(2^128) multiply on a uint64[2] (lo, hi) scalar pair — for the
    verify sumcheck replay, which runs on host numpy scalars (not device)."""
    ag = np.asarray(a, np.uint64).view(ghash._GHASH_HOST)
    bg = np.asarray(b, np.uint64).view(ghash._GHASH_HOST)
    return np.asarray(ag * bg).view(np.uint64)


def _root_f128(root):
    """flock root_to_f128: F128 from the FIRST 16 bytes of the 32-byte root."""
    return root[:16].view(np.uint64)


@functools.partial(frx.jit, static_argnums=(4, 5, 6))
def _replay_round_fs(t, msgs_g, post_rb_g, commits_g, log_batch_size, arities,
                     num_epochs):
    """The verify sumcheck's Fiat-Shamir replay as ONE device program: per round
    observe (u0, u2) and sample r, with the prover's post-row-batch / per-epoch
    commitment observes interleaved on the same static schedule. Returns the
    challenge stack; byte-identical to the op-by-op replay."""
    challenges = []
    current_epoch = rounds_in_epoch = 0
    for rnd in range(msgs_g.shape[0]):
        t = t.observe_scalar(msgs_g[rnd, 0]).observe_scalar(msgs_g[rnd, 1])
        t, r = t.sample_scalar()
        challenges.append(r)
        if rnd + 1 == log_batch_size and arities:
            t = t.observe_scalar(post_rb_g)
        if rnd >= log_batch_size:
            rounds_in_epoch += 1
            if rounds_in_epoch == arities[current_epoch]:
                if current_epoch + 1 < num_epochs:
                    t = t.observe_scalar(commits_g[current_epoch])
                rounds_in_epoch = 0
                current_epoch += 1
    return t, jnp.stack(challenges)


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
    """flock's BaseFold verifier (`pcs::basefold::verify`), authored in frx. Replays
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

    log_msg = len(proof.round_messages)
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

    if len(proof.round_commitments) != num_fri_commits:
        return reject("InvalidProofShape:round_commitments")
    # #queries is a soundness parameter, not a prover choice (flock SECURITY note).
    if len(proof.queries) != fri.default_fri_queries(log_inv_rate):
        return reject("InvalidProofShape:n_queries")

    # ---- replay sumcheck + observe commitments in lockstep with the prover ----
    # One jitted device program for the whole FS replay (the schedule is static
    # given the proof shape); the running-target algebra replays on host after,
    # off the materialized challenges — it reads the transcript nowhere.
    # to_lanes on read: the proof holds ghash round messages, but this verifier
    # also runs on uint64-lane test proofs, so it stays dtype-agnostic.
    msgs = np.stack([(ghash.to_lanes(u0), ghash.to_lanes(u2))
                     for u0, u2 in proof.round_messages])            # [log_msg, 2, 2] lanes
    if arities:
        post_rb_g = ghash.to_ghash(
            jnp.asarray(_root_f128(np.asarray(proof.post_row_batch_commit)).copy()))
        commits = np.stack(
            [_root_f128(np.asarray(c)).copy() for c in proof.round_commitments]
        ) if num_fri_commits else np.zeros((0, 2), np.uint64)
    else:
        post_rb_g = ghash.to_ghash(jnp.zeros(2, jnp.uint64))          # unused (static)
        commits = np.zeros((0, 2), np.uint64)
    ch._t, ch_stack = _replay_round_fs(
        ch._t, ghash.to_ghash(jnp.asarray(msgs)), post_rb_g,
        ghash.to_ghash(jnp.asarray(commits)), log_batch_size, tuple(arities),
        num_epochs)
    challenges = list(ghash.from_ghash_host(ch_stack))                # [log_msg] of [2]

    running_target = np.asarray(target, np.uint64)
    for rnd in range(log_msg):
        u0, u2 = msgs[rnd]
        r = challenges[rnd]
        # running_target = u0 + r·(running_target + u2) + r²·u2   (flock).
        u1 = running_target ^ u2
        rr = _hf_mul(r, r)
        running_target = (u0 ^ _hf_mul(r, u1)) ^ _hf_mul(rr, u2)

    info = {"reason": "", "challenges": challenges}

    # ---- final sumcheck + codeword-constancy checks (lanes: the FRI fold
    # kernels below are lane-based, so the fold-consistency stays uint64) ----
    final_a = ghash.to_lanes(proof.final_a)
    final_b = ghash.to_lanes(proof.final_b)
    if not np.array_equal(_hf_mul(final_a, final_b), running_target):
        return reject("SumcheckFinalMismatch", challenges)
    final_cw = ghash.to_lanes(proof.final_codeword)
    if final_cw.shape[0] != (1 << log_inv_rate):
        return reject("FinalCodewordNotConstant:len", challenges)
    if not np.all([np.array_equal(final_cw[i], final_cw[0]) for i in range(final_cw.shape[0])]):
        return reject("FinalCodewordNotConstant", challenges)
    if not np.array_equal(final_cw[0], final_a):
        return reject("SumcheckFriMismatch", challenges)

    # ---- resample query positions (challenger state matches prover) ----
    n_q = len(proof.queries)
    ch._t, pos_g = fs.sample_chain(ch._t, n_q)
    positions = (ghash.from_ghash_host(pos_g)[:, 0].astype(np.int64)
                 & ((1 << k_code) - 1))

    arity_0 = arities[0] if arities else 0
    ch_g = [ghash.to_ghash(r.reshape(1, 2)).reshape(()) for r in challenges]  # ghash betas

    # Gather per-query leaves (uint64 SoA) once.
    q_pos = np.array([q[0] for q in proof.queries], dtype=np.int64)
    if not np.array_equal(q_pos, positions):
        return reject("FoldMismatch:position", challenges)
    init_leaves = np.stack([np.asarray(q[1], np.uint64) for q in proof.queries])  # [Q, num_ntts, 2]
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
        post_rb_leaves = np.stack([np.asarray(q[2], np.uint64) for q in proof.queries])  # [Q, 2^a0, 2]
        inner = (positions & ((1 << arity_0) - 1))
        got = post_rb_leaves[np.arange(n_q), inner]  # [Q, 2]
        if not np.array_equal(got, prbv):
            return reject("FoldMismatch:t2_crosscheck", challenges)
        coset_g = ghash.to_ghash(post_rb_leaves.reshape(-1, 2)).reshape(n_q, 1 << arity_0)
        expected_g = _fold_coset(code, coset_g,
                                 ch_g[log_batch_size:log_batch_size + arity_0],
                                 k_code, positions >> arity_0, k_code)
        expected = ghash.from_ghash_host(expected_g)

        cum = arity_0
        for i in range(num_fri_commits):
            next_arity = arities[i + 1]
            leaves_i = np.stack([np.asarray(q[3][i], np.uint64) for q in proof.queries])  # [Q, 2^na, 2]
            if leaves_i.shape[1] != (1 << next_arity):
                return reject("InvalidProofShape:epoch_leaf_len", challenges)
            p_at = positions >> cum
            offset = p_at & ((1 << next_arity) - 1)
            got = leaves_i[np.arange(n_q), offset]  # [Q, 2]
            if not np.array_equal(got, expected):
                return reject(f"FoldMismatch:epoch{i}", challenges)
            leaf_g = ghash.to_ghash(leaves_i.reshape(-1, 2)).reshape(n_q, 1 << next_arity)
            expected_g = _fold_coset(code, leaf_g,
                                     ch_g[log_batch_size + cum:log_batch_size + cum + next_arity],
                                     k_code - cum, p_at >> next_arity, k_code)
            expected = ghash.from_ghash_host(expected_g)
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

    legs.append(_leg(np.asarray(proof.initial_multi_proof), 1 << k_code,
                     positions, init_leaves, initial_codeword_root))
    if arities:
        legs.append(_leg(np.asarray(proof.post_row_batch_multi_proof),
                         1 << (k_code - arity_0), positions >> arity_0,
                         post_rb_leaves, proof.post_row_batch_commit))
        cum = arity_0
        for i in range(num_fri_commits):
            next_arity = arities[i + 1]
            leaves_i = np.stack([np.asarray(q[3][i], np.uint64) for q in proof.queries])
            leaf_idx = (positions >> cum) >> next_arity
            legs.append(_leg(np.asarray(proof.epoch_multi_proofs[i]),
                             1 << (k_code - cum - next_arity), leaf_idx,
                             leaves_i, proof.round_commitments[i]))
            cum += next_arity

    if not merkle.verify_openings(legs):
        return reject("MerkleFailed", challenges)

    return True, info
