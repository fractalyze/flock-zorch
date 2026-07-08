"""flock instantiation of zorch's BaseFold choreography + kernel seams
(flock-zorch#50) — the flock side of `zorch.pcs.basefold`'s non-native fold
cadence, byte-identical to the in-tree `basefold.prove`.

`zorch.pcs.basefold` runs a generic interleaved sumcheck + fold whose per-round
FS framing is a `BasefoldChoreography`, whose per-round algebra is a
`SumcheckKernel`, and whose fold schedule is a `BasefoldConfig`. flock's BaseFold
is a degree-2 product sumcheck over `(a, b)` with the message `(u0, u2)`, a
deferred row-batch over the first `log_batch_size` rounds, then multi-arity FRI
epochs with octopus multi-proofs and `root_to_f128` (FIRST 16 bytes) root
observes. This module supplies those three flock deltas plus the driver:

- `FlockProductKernel`: the `(a, b)` product round — message
  `(Σaₑbₑ, Σ(aₑ+aₒ)(bₑ+bₒ))`, folding both vectors by `fold_single` (the char-2
  multilinear bind), terminal `(a[0], b[0])`.
- `FlockBasefoldChoreography`: the `flock-basefold-v0` label bind, eager `(u0,u2)`
  emission (two scalar observes), `root_to_f128` root observes, NO terminal
  observe (flock samples queries straight off the last challenge), and flock's
  per-query masked query positions (one scalar squeeze each, dups kept, order
  kept — NOT distinct/sorted).
- `flock_basefold_config`: the row-batch-prefix + `compute_fri_arities` schedule.
- `prove_flock_basefold`: bridge `ch._t` into a `FlockTranscript`, drive
  `open_with_basis`, and assemble flock's BaseFoldProof dict (octopus via
  `merkle.paths_to_multi_proof`), writing `ch._t` back.

The codeword fold rides zorch's `AdditiveReedSolomon.fold` on the native
`binary_field_ghash` dtype (its LCH twiddles byte-match flock's FRI, #40), and
the commits ride `merkle.GHASH_TREE` (SHA-256 over the raw F128 leaf bytes, same
preimage as flock's `merkle_tree`). Reuses `zorch_ligerito`'s `FlockTranscript`
bridge and F128 serde. Requires `jax_enable_x64`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from zorch.coding.additive_reed_solomon import AdditiveReedSolomon
from zorch.pcs.basefold.choreography import BasefoldChoreography
from zorch.pcs.basefold.config import BasefoldConfig
from zorch.pcs.basefold.kernel import SumcheckKernel
from zorch.pcs.basefold.prover import BasefoldProver, BasefoldProverData

from flock_zorch import field, fri, merkle
from flock_zorch.zorch_ligerito import FlockTranscript

FLOCK_BASEFOLD_LABEL = b"flock-basefold-v0"


# --- the (a, b) product sumcheck kernel --------------------------------------


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
    (the flock verifier is untouched by #50)."""

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
        return FlockTranscript(transcript.inner.observe_label(FLOCK_BASEFOLD_LABEL))

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
        # flock's BaseFold queries: one scalar F128 squeeze per query, low limb
        # mod block_len; dups kept, order kept (NOT distinct, NOT sorted).
        inner = transcript.inner
        out: list[int] = []
        for _ in range(count):
            inner, buf = inner.sample_scalar(16)
            out.append(int.from_bytes(buf[:8], "little") % block_len)
        return FlockTranscript(inner), jnp.asarray(out, jnp.int32)


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


# --- wire assembly: zorch CadenceProof -> flock's BaseFoldProof dict ----------


def _lohi_scalar(x: Array) -> np.ndarray:
    """ghash scalar -> uint64 [2] (lo, hi), flock's F128 representation."""
    b = np.asarray(lax.bitcast_convert_type(x, jnp.uint8)).tobytes()
    return np.frombuffer(b, np.uint64).copy()


def _lohi(x: Array) -> np.ndarray:
    """ghash (any shape) -> (-1, 2) uint64 lo‖hi (the uint8 bitcast direction)."""
    b = np.asarray(lax.bitcast_convert_type(x, jnp.uint8)).tobytes()
    return np.frombuffer(b, np.uint64).reshape(-1, 2).copy()


def _leaf_rows(row: Array) -> np.ndarray:
    """opened leaf rows ghash [Q, w] -> uint64 [Q, w, 2] (per-query F128 leaf)."""
    q, w = int(row.shape[0]), int(row.shape[1])
    return _lohi(row).reshape(q, w, 2)


def _octopus(opening, num_leaves: int, positions: Array) -> np.ndarray:
    """flock's octopus multi-proof from a zorch `Opening`'s per-query paths +
    the sampled positions (byte-identical to `merkle.merkle_multi_proof`)."""
    paths = np.stack([np.asarray(p, np.uint8) for p in opening.path], axis=1)
    return merkle.paths_to_multi_proof(paths, num_leaves, np.asarray(positions))


def _flock_basefold_dict(proof, k_code: int, log_inv_rate: int, num_ntts: int) -> dict:
    """zorch `CadenceProof` -> flock's `BaseFoldProof` dict, byte-identical to
    the in-tree `basefold.prove` return."""
    arities = fri.compute_fri_arities(k_code - log_inv_rate)
    num_fri_commits = max(len(arities) - 1, 0)
    has_prefix_commit = bool(arities)  # the post-row-batch commit exists iff FRI does

    round_messages = [
        (_lohi_scalar(u0), _lohi_scalar(u2)) for (u0, u2) in proof.round_messages
    ]
    roots = [np.asarray(r, np.uint8).copy() for r in proof.commit_roots]
    post_rb_root = roots[0] if has_prefix_commit else np.zeros(32, np.uint8)
    round_commitments = roots[1:]  # per-epoch roots (all but the last epoch)

    final_a = _lohi_scalar(proof.final_state[0])
    final_b = _lohi_scalar(proof.final_state[1])
    final_codeword = _lohi(proof.final_codeword)  # [2^log_inv_rate, 2]

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

    return {
        "round_messages": round_messages,
        "post_row_batch_commit": post_rb_root,
        "round_commitments": round_commitments,
        "final_a": final_a,
        "final_b": final_b,
        "final_codeword": final_codeword,
        "queries": queries,
        "initial_multi_proof": initial_mp,
        "post_row_batch_multi_proof": post_rb_mp,
        "epoch_multi_proofs": epoch_mps,
    }


def prove_flock_basefold(
    z_packed,
    b_combined,
    codeword,
    initial_tree,
    k_code,
    log_inv_rate,
    log_batch_size,
    n_queries,
    ch,
) -> dict:
    """Drive `zorch.pcs.basefold` over flock's shared challenger and assemble a
    flock `BaseFoldProof` dict — byte-identical to the retired in-tree
    `basefold.prove`.

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

    a = field.to_ghash(jnp.asarray(z_packed))  # [2^log_msg]
    b = field.to_ghash(jnp.asarray(b_combined))  # [2^log_msg]
    cw_g = field.to_ghash(jnp.asarray(codeword))  # [n_pos * num_ntts]

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
    value = field.to_ghash(jnp.zeros(2, jnp.uint64))  # ignored (no target on the wire)

    proof, t = prover.open_with_basis(pd, b, value, FlockTranscript(ch._t))
    ch._t = t.inner
    return _flock_basefold_dict(proof, k_code, log_inv_rate, num_ntts)
