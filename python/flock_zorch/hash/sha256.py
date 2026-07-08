"""SHA-256 over uint32 lanes — flock-zorch's hash for `merkle.rs` leaves/levels and
the Fiat-Shamir challenger. Byte-identical to flock's `Sha256::digest` (the `sha2`
crate), which is the FIPS 180-4 standard, so the core (round constants, message
schedule, compression, host pad, digest) is reused verbatim from `zorch.hash.sha256`
— the scheme-agnostic spine — rather than duplicated here. This module is just that
re-export plus the flock oracle anchor below (the device-resident Merkle pad now
lives in `merkle.py`).

The flock-side oracle gate (`testing/sha256_oracle_test.py`) still pins `_pad` /
`_digest_words` (by attribute, here) to flock's golden bytes, so the reused core
stays anchored to upstream flock even though the implementation lives in zorch.

Requires no x64; all arithmetic is uint32 (wraps mod 2^32 in XLA).
"""
from __future__ import annotations

# Reuse the byte-identical SHA-256 core from zorch (the scheme-agnostic spine).
# Re-exported so merkle and the flock oracle gate can reach them as sha256.<name>.
# `_digest_words_marked` is the same digest wrapped in the name-routed
# `zorch.sha256` composite (inlines its decomposition until an emitter is wired,
# so bytes are unchanged) — the Merkle leaf/level path hashes through it.
from zorch.hash.sha256 import (  # noqa: F401
    U32, _K, _H0, _Kd, _rotr, _pad, _compress, _digest_words,
    _digest_words_marked, digest,
)
