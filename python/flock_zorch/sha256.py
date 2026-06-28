"""SHA-256 over uint32 lanes — flock-zorch's hash for `merkle.rs` leaves/levels and
the Fiat-Shamir challenger. Byte-identical to flock's `Sha256::digest` (the `sha2`
crate), which is the FIPS 180-4 standard, so the core (round constants, message
schedule, compression, host pad, digest) is reused verbatim from `zorch.hash.sha256`
— the scheme-agnostic spine — rather than duplicated here. Only `_pad_device`, the
all-jnp padding for the device-resident Merkle tree, is flock-local.

The flock-side oracle gate (`testing/sha256_oracle_test.py`) still pins `_pad` /
`_digest_words` (by attribute, here) to flock's golden bytes, so the reused core
stays anchored to upstream flock even though the implementation lives in zorch.

Requires no x64; all arithmetic is uint32 (wraps mod 2^32 in XLA).
"""
from __future__ import annotations

import jax.numpy as jnp

# Reuse the byte-identical SHA-256 core from zorch (the scheme-agnostic spine).
# Re-exported so merkle and the flock oracle gate can reach them as sha256.<name>.
from zorch.hash.sha256 import (  # noqa: F401
    U32, _K, _H0, _Kd, _rotr, _pad, _compress, _digest_words, digest,
)


def _pad_device(msg, length: int):
    """Device SHA-256 pad: jnp uint8 [B, length] -> uint32 [B, nblocks, 16] BE.

    Same layout as `_pad` but all-jnp (no host round-trip), for the Merkle tree
    where nodes stay device-resident across levels. `length` is static. flock-local;
    the compression itself is the reused `_digest_words`.
    """
    b = msg.shape[0]
    bitlen = length * 8
    nblocks = (length + 8) // 64 + 1
    total = nblocks * 64
    padded = jnp.zeros((b, total), dtype=jnp.uint8)
    padded = padded.at[:, :length].set(msg)
    padded = padded.at[:, length].set(jnp.uint8(0x80))
    for i in range(8):  # 8-byte big-endian bit length at the tail (static bytes)
        padded = padded.at[:, total - 8 + i].set(jnp.uint8((bitlen >> (8 * (7 - i))) & 0xFF))
    words = padded.reshape(b, nblocks, 16, 4).astype(jnp.uint32)
    return (words[..., 0] << U32(24)) | (words[..., 1] << U32(16)) | (words[..., 2] << U32(8)) | words[..., 3]
