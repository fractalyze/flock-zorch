"""PCS commit, authored in jax — byte-identical to flock's `pcs::commit`.

The first FULL sub-protocol with a byte-serializable output (a 32-byte Merkle
root). Construction (flock `pcs/commit.rs`):

    z_packed ──► zero-pad to 2^k_code positions ──► interleaved forward NTT
              ──► codeword (SoA, position-major) ──► SHA-256 Merkle ──► root

With `(m, log_inv_rate, log_batch_size)`:
  log_msg = m - 7,  log_dim = log_msg - log_batch_size,  k_code = log_dim + log_inv_rate,
  num_ntts = 2^log_batch_size. Each Merkle leaf is one codeword position =
  num_ntts F128 = num_ntts*16 bytes. In the paper's App C.1 notation: rate
  ρ = 2^-log_inv_rate; message length = log_msg = m - LOG_PACKING (the 7-bit F128
  packing offset); the interleaved lanes (columns) are the Merkle leaves.

This equals flock's definitional encoding (zero-pad + full interleaved NTT, the
oracle flock's own `commit_matches_full_ntt_oracle` test pins); flock's
replicate-fill / start-at-layer-`log_inv_rate` is just a perf shortcut for the
same codeword. The NTT inherits clmad on GPU; Merkle is a <1% tail.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from flock_zorch import field, ntt as ntt_mod, merkle, sha256  # noqa: F401  (sha256 via merkle)

LOG_PACKING = 7


def pack_witness(z_bits: np.ndarray, m: int) -> np.ndarray:
    """Pack a Boolean witness (uint8/bool [2^m]) into F128 [2^(m-7), 2] uint64.

    bit r of out[i] = z[i*128 + r] (little-endian within the 128-bit element),
    matching flock's `pack::pack_witness`.
    """
    z = np.asarray(z_bits, dtype=np.uint64).reshape(-1, 128)  # [n_packed, 128]
    weights = (np.uint64(1) << np.arange(64, dtype=np.uint64))  # [64]
    lo = (z[:, :64] * weights).sum(axis=1, dtype=np.uint64)
    hi = (z[:, 64:] * weights).sum(axis=1, dtype=np.uint64)
    return np.stack([lo, hi], axis=1)  # [n_packed, 2]


def _encode_codeword(z_packed, m: int, log_inv_rate: int, log_batch_size: int, mul=field.mul):
    """z_packed -> (codeword uint64 [2^k_code · num_ntts, 2], n_pos_code, num_ntts).
    Zero-pad to 2^k_code positions, interleaved forward NTT (the shared commit/open
    encode). Used by both `commit_root` and `commit`."""
    log_msg = m - LOG_PACKING
    log_dim = log_msg - log_batch_size
    k_code = log_dim + log_inv_rate
    num_ntts = 1 << log_batch_size
    n_pos_msg = 1 << log_dim
    n_pos_code = 1 << k_code

    x = jnp.asarray(z_packed).reshape(n_pos_msg, num_ntts, 2)
    pad = jnp.zeros((n_pos_code - n_pos_msg, num_ntts, 2), dtype=x.dtype)
    codeword = jnp.concatenate([x, pad], axis=0).reshape(n_pos_code * num_ntts, 2)
    tw = jnp.asarray(ntt_mod.compute_twiddles(k_code))
    codeword = ntt_mod.forward_transform_interleaved(codeword, tw, k_code, num_ntts, mul=mul)
    return codeword, n_pos_code, num_ntts


def commit(z_packed, m: int, log_inv_rate: int, log_batch_size: int, mul=field.mul,
           use_host_sha: bool = False):
    """Full PCS commit: returns (root uint8[32], codeword uint64[.,2], tree uint8[2n-1,32]).
    The codeword + tree are the prover_data the PCS open consumes. Byte-identical
    to flock `pcs::commit` (root) + its ProverData (codeword, merkle_tree)."""
    codeword, n_pos_code, num_ntts = _encode_codeword(z_packed, m, log_inv_rate, log_batch_size, mul)
    cw_np = np.asarray(codeword)
    leaves = cw_np.reshape(n_pos_code, num_ntts * 2).view(np.uint8)
    tree = merkle.merkle_tree(leaves, use_host_sha=use_host_sha)
    return tree[-1], cw_np, tree


def commit_root(z_packed, m: int, log_inv_rate: int, log_batch_size: int, mul=field.mul,
                use_host_sha: bool = False) -> np.ndarray:
    """32-byte Merkle root of the PCS commitment to `z_packed`.

    z_packed: uint64 [2^(m-7), 2]. Returns uint8 [32], byte-identical to
    `flock::pcs::commit(z_packed, params).root`.
    """
    log_msg = m - LOG_PACKING
    log_dim = log_msg - log_batch_size
    k_code = log_dim + log_inv_rate
    num_ntts = 1 << log_batch_size
    n_pos_msg = 1 << log_dim
    n_pos_code = 1 << k_code

    # SoA: z_packed flat = codeword[pos*num_ntts + lane] for the first 2^log_dim
    # positions; zero-pad the remaining positions up to 2^k_code.
    x = jnp.asarray(z_packed).reshape(n_pos_msg, num_ntts, 2)
    pad = jnp.zeros((n_pos_code - n_pos_msg, num_ntts, 2), dtype=x.dtype)
    codeword = jnp.concatenate([x, pad], axis=0).reshape(n_pos_code * num_ntts, 2)

    tw = jnp.asarray(ntt_mod.compute_twiddles(k_code))
    codeword = ntt_mod.forward_transform_interleaved(codeword, tw, k_code, num_ntts, mul=mul)

    # Each leaf = one position's num_ntts F128 = num_ntts*16 LE bytes (F128 is
    # lo||hi little-endian, same as a uint64 array viewed as bytes on x86).
    leaves = np.asarray(codeword).reshape(n_pos_code, num_ntts * 2).view(np.uint8)
    return merkle.merkle_root(leaves, use_host_sha=use_host_sha)
