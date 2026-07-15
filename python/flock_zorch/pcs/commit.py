"""PCS commit, authored in frx — byte-identical to flock's `pcs::commit`.

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
same codeword. The encode runs on the native `binary_field_ghash` dtype (its
`lax.ntt` lowers to the GPU clmul kernel); Merkle is a <1% tail.
"""
from __future__ import annotations

import numpy as np
import frx.numpy as jnp
from frx import lax

from zorch.coding.additive_reed_solomon import AdditiveReedSolomon

from flock_zorch import field
from flock_zorch.hash import merkle

LOG_PACKING = field.LOG_PACKING


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


def _encode_codeword(z_packed, m: int, log_inv_rate: int, log_batch_size: int):
    """z_packed -> (codeword uint64 [2^k_code · num_ntts, 2], n_pos_code, num_ntts).
    RS-encode each row-batch lane with zorch's `coding.AdditiveReedSolomon` over
    `binary_field_ghash` (`lax.ntt` dispatches the additive-NTT LCH transform,
    zero-padding message positions to 2^k_code — byte-identical to flock's
    `forward_transform_interleaved`). Shared by `commit_root` and `commit`."""
    log_msg = m - LOG_PACKING
    log_dim = log_msg - log_batch_size
    k_code = log_dim + log_inv_rate
    num_ntts = 1 << log_batch_size
    n_pos_msg = 1 << log_dim
    n_pos_code = 1 << k_code

    # z_packed is SoA position-major with num_ntts interleaved lanes
    # (z_packed[pos*num_ntts + lane]): reinterpret to ghash, RS-encode each lane,
    # then transpose back to the SoA layout. field.{to,from}_ghash route the
    # ghash<->uint64 bitcasts through uint32 lanes, so the codeword stays
    # device-resident (the direct ghash->uint64 bitcast returns zeros, zorch#399).
    code = AdditiveReedSolomon(n_pos_msg, 1 << log_inv_rate, jnp.binary_field_ghash)
    msg = field.to_ghash(jnp.asarray(z_packed).reshape(n_pos_msg, num_ntts, 2))
    cw = code.encode(msg.T)  # [num_ntts, n_pos_code]
    codeword = field.from_ghash(cw.T).reshape(n_pos_code * num_ntts, 2)
    return codeword, n_pos_code, num_ntts


def _codeword_leaves(codeword, n_pos_code: int, num_ntts: int):
    """Device SoA codeword uint64 [n_pos_code·num_ntts, 2] -> Merkle leaves uint8
    [n_pos_code, num_ntts·16]. Each leaf = one codeword position's num_ntts F128 =
    num_ntts·16 LE bytes (F128 is lo‖hi little-endian, i.e. the uint64 lanes viewed
    as bytes). The uint64->uint8 bitcast is device-resident and correct on both
    backends, so `merkle` receives device leaves with no host round-trip."""
    return lax.bitcast_convert_type(
        codeword.reshape(n_pos_code, num_ntts * 2), jnp.uint8
    ).reshape(n_pos_code, num_ntts * 16)


def commit(z_packed, m: int, log_inv_rate: int, log_batch_size: int):
    """Full PCS commit: returns (root uint8[32], codeword uint64[.,2] device, tree
    uint8[2n-1,32]). The codeword + tree are the prover_data the PCS open consumes.
    Byte-identical to flock `pcs::commit` (root) + its ProverData (codeword,
    merkle_tree). Encode + Merkle leaves stay device-resident; the codeword stays on
    device (BaseFold folds and gathers it there), only the 32-byte tree crosses to host."""
    codeword, n_pos_code, num_ntts = _encode_codeword(z_packed, m, log_inv_rate, log_batch_size)
    tree = merkle.merkle_tree(_codeword_leaves(codeword, n_pos_code, num_ntts))
    return tree[-1], codeword, tree


def commit_root(z_packed, m: int, log_inv_rate: int, log_batch_size: int) -> np.ndarray:
    """32-byte Merkle root of the PCS commitment to `z_packed`.

    z_packed: uint64 [2^(m-7), 2]. Returns uint8 [32], byte-identical to
    `flock::pcs::commit(z_packed, params).root`. Device-resident end to end
    (encode -> Merkle); only the 32-byte root crosses to host.
    """
    codeword, n_pos_code, num_ntts = _encode_codeword(z_packed, m, log_inv_rate, log_batch_size)
    return merkle.merkle_root(_codeword_leaves(codeword, n_pos_code, num_ntts))
