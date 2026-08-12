# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BLAKE3 Fiat-Shamir challenger — the flock-challenge benchmark profile's FS arm.

The ranked benchmark (Layr-Labs/flock-challenge) pins its transcript hash to
BLAKE3: the worker and the harness verifier both run
`FsChallenger::with_hash(domain, Blake3)`, so a proof aimed at their window has
to draw every challenge from this arm. Its framing is the same Merlin-over-hash
duplex the SHA-256 challenger speaks — identical op tags, length prefixes, and
squeeze re-absorb, i.e. zorch's `ByteHashTranscript` wire — with exactly two
per-hash divergences (both read from the fork's `challenger.rs`):

- **The squeeze is an XOF read, not a counter chain.** SHA-256 has fixed
  output, so its challenge stream is derived as `SHA256(state ‖ ctr_le8)` for
  ctr = 0, 1, …, 32 bytes a block. BLAKE3 *is* an XOF: the fork finalizes the
  absorbed state once and reads the stream straight out, so no counter bytes
  ever enter the transcript and even a 16-byte scalar draw differs from the
  counter construction.
- **The PoW pre-image is one whole 64-byte block.** `state_digest ‖ nonce_le8`
  is zero-padded from 40 to 64 bytes (a whole-block single-chunk message is
  what BLAKE3's batched kernels want); the SHA-256 arm's pre-image stays 40.

The substrate is hash-frx's host BLAKE3 (`HostBlake3`): the Fiat-Shamir chain
is strictly sequential and every draw is read back immediately, which is the
case the host rows exist for. This challenger is therefore the
byte-correctness reference for the benchmark profile — the twin a device
BLAKE3 transcript would be pinned against, the same way zorch pins
`Sha256FieldTranscript` against `ByteHashTranscript` — and it is NOT a drop-in
for the jitted prove path, which threads a device transcript pytree through
its round bodies.

Byte-gated against transcripts dumped from the fork (d866043) by
`testing/blake3_challenger_test.py`, which also carries the dump recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hash_frx.blake3.byte_hashes import HostBlake3
from zorch.byte_transcript import (
    ByteHashTranscript,
    _leading_zero_bits_ok,
    _len8,
    _validate_pow_bits,
)

from flock_zorch.ghash import _ghash_to_lanes, _lanes_to_ghash

_F128_BYTES = 16
# The fork's BLAKE3 PoW pre-image length: state digest (32) ‖ nonce (8) ‖ zero
# padding to one whole block.
_POW_BLOCK = 64
# Nonces per batched digest call in the grind. Purely a host-loop amortization
# knob — windows are scanned in ascending order and the first hit inside a
# window is the window's smallest, so the result is the globally smallest
# nonce at any window size.
_GRIND_WINDOW = 1 << 12


@dataclass(frozen=True)
class Blake3ByteTranscript(ByteHashTranscript):
    """`ByteHashTranscript` with the fork's two BLAKE3 arms: XOF squeeze and
    whole-block PoW pre-image.

    The wire framing is inherited — this class pins the base's `_absorb` /
    `_squeeze` extension points plus the two PoW entry points, and nothing
    else. The fork-fixture gate in `testing/blake3_challenger_test.py` catches
    a zorch bump that moves those seams.
    """

    def _absorb(self, payload: bytes) -> Blake3ByteTranscript:
        # The base constructs `ByteHashTranscript` literally rather than
        # `type(self)`, so absorbing through it would decay to the counter
        # squeeze on the very first observe.
        return Blake3ByteTranscript(self.buffer + payload, self.byte_hash)

    def _squeeze(self, n: int) -> bytes:
        # XOF read: finalize the absorbed stream once and take `n` bytes.
        # hash-frx's BLAKE3 rows read the extendable output through their
        # `output_size`, so a digest at length `n` IS the XOF read.
        if n <= 0:
            return b""
        row = np.frombuffer(self.buffer, dtype=np.uint8)[None, :]
        return bytes(np.asarray(HostBlake3(n).digest(row))[0])

    def _pow_preimages(self, state_digest: bytes, base: int, count: int) -> np.ndarray:
        """uint8 `[count, 64]` whole-block pre-images for nonces `base ..
        base + count - 1`: `state_digest ‖ nonce_le8 ‖ zero padding`."""
        rows = np.zeros((count, _POW_BLOCK), dtype=np.uint8)
        rows[:, :32] = np.frombuffer(state_digest, dtype=np.uint8)
        nonces = base + np.arange(count, dtype=np.uint64)
        rows[:, 32:40] = nonces.astype("<u8").view(np.uint8).reshape(count, 8)
        return rows

    def _grind(self, state_digest: bytes, bits: int) -> int:
        # Same contract as the base: the lowest u64 nonce whose PoW digest has
        # `bits` leading zero bits — only the pre-image layout differs.
        base = 0
        while True:
            rows = self._pow_preimages(state_digest, base, _GRIND_WINDOW)
            digests = np.asarray(self.byte_hash.digest(rows))
            hits = np.flatnonzero(_leading_zero_bits_ok(digests, bits))
            if hits.size:
                return base + int(hits[0])
            base += _GRIND_WINDOW

    def verify_pow(self, nonce: int, *, bits: int) -> tuple[Blake3ByteTranscript, bool]:
        _validate_pow_bits(bits, self.byte_hash.digest_size)
        if bits == 0:
            ok = nonce == 0
        else:
            row = self._pow_preimages(self._digest(), nonce, 1)
            ok = bool(
                _leading_zero_bits_ok(np.asarray(self.byte_hash.digest(row)), bits)[0]
            )
        return self.observe_bytes(_len8(nonce)), ok


class Blake3Challenger:
    """Mutable wrapper over the functional BLAKE3 byte transcript, mirroring
    flock-core's `&mut self` `FsChallenger::with_hash(domain, Blake3)` — the
    same API surface `sha256_challenger.Sha256Challenger` gives the SHA-256 transcript.

    Observes and samples carry `binary_field_ghash` elements (host numpy — the
    transcript is a host object); a scalar draw and a slice draw frame
    differently on the wire, so a length-1 vector still passes an explicit
    `n=1`, exactly as on the SHA-256 arm.
    """

    def __init__(self, domain: bytes):
        self._t = Blake3ByteTranscript.new(bytes(domain), HostBlake3())

    def observe_label(self, label: bytes) -> None:
        self._t = self._t.observe_label(bytes(label))

    def observe_bytes(self, data) -> None:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            data = np.asarray(data, dtype=np.uint8).tobytes()
        self._t = self._t.observe_bytes(bytes(data))

    def observe_f128(self, g) -> None:
        """Observe F128 (`binary_field_ghash`) — a scalar or a 1-D slice,
        framed by shape (flock scalar-frames a single element, slice-frames
        many)."""
        a = np.asarray(g)
        if a.ndim > 1:
            raise ValueError(
                f"observe_f128 takes a scalar or a 1-D slice, got {a.shape}"
            )
        payload = np.ascontiguousarray(_ghash_to_lanes(a), dtype=np.uint64).tobytes()
        if a.ndim == 0:
            self._t = self._t.observe_scalar(payload)
        else:
            self._t = self._t.observe_slice(payload, a.shape[0])

    def sample_f128(self, n: int | None = None):
        """Sample F128 as `binary_field_ghash`. Bare `sample_f128()` is a single
        scalar draw; `sample_f128(n)` is a length-`n` slice — the two frame
        differently on the wire (scalar vs slice(1) are NOT the same bytes)."""
        if n is None:
            self._t, buf = self._t.sample_scalar(_F128_BYTES)
            return _lanes_to_ghash(np.frombuffer(buf, dtype=np.uint64))
        self._t, buf = self._t.sample_slice(n, _F128_BYTES)
        return _lanes_to_ghash(np.frombuffer(buf, dtype=np.uint64).reshape(n, 2))

    def grind_pow(self, bits: int) -> int:
        self._t, nonce = self._t.grind_pow(bits)
        return int(nonce)

    def verify_pow(self, nonce: int, *, bits: int) -> bool:
        """Verifier mirror of `grind_pow`: check, then absorb the nonce
        REGARDLESS so the transcript stays in lockstep with the prover's."""
        self._t, ok = self._t.verify_pow(int(nonce), bits=bits)
        return bool(ok)
