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

Two rows ship, mirroring what the SHA-256 arm gets from zorch:

- `Blake3ByteTranscript` / `Blake3Challenger` — the host reference, over
  hash-frx's `HostBlake3`. zorch's `ByteHashTranscript` fills this role for
  SHA-256; the fork's two BLAKE3 deviations are not upstream, so flock hosts
  its own. This is the byte oracle the device row is pinned against.
- `Blake3DeviceChallenger` — the prove path: `Sha256Challenger`'s surface over
  the device `Blake3FieldTranscript`, whose state is a fixed-shape pytree, so
  the four transcript-threading jitted zones (zerocheck ML ladder, lincheck
  prove_inf_product, ring-switch reduce, the Ligerito open) carry it through
  their loops instead of de-compiling into host loops.

Byte-gated against transcripts dumped from the fork (d866043) by
`testing/blake3_challenger_test.py`, which also carries the dump recipe;
`testing/blake3_field_transcript_test.py` pins the device row against the host
one, op for op.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from hash_frx.blake3.byte_hashes import HostBlake3
from zorch.byte_transcript import (
    ByteHashTranscript,
    _leading_zero_bits_ok,
    _len8,
    _validate_pow_bits,
)

from flock_zorch import fs
from flock_zorch.ghash import _ghash_to_lanes, _lanes_to_ghash
from flock_zorch.hash.blake3_field_transcript import Blake3FieldTranscript

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


@functools.lru_cache(maxsize=None)
def _initial_device_transcript(domain: bytes):
    """Memoize the seeded device state per domain.

    `Blake3FieldTranscript.new` absorbs the domain through the full device
    absorb program — 0.5 s of trace per construction. Array values are
    immutable and every challenger replaces `_t` rather than mutating it, so
    one seeded state is safe to share between proves. `Sha256Challenger` does
    the same for the same reason.
    """
    return Blake3FieldTranscript.new(domain, fnp.binary_field_ghash)


class Blake3DeviceChallenger:
    """`Sha256Challenger`'s surface over the device transcript — the BLAKE3
    profile's prove-path challenger, and the structural twin of the SHA-256 one.

    `Blake3FieldTranscript`'s state is a fixed-shape pytree, so a jitted round
    loop carries it and the sumcheck loop stays inside the compiled program.
    (`Blake3Challenger` cannot: a host transcript is not a pytree, and a prove
    driven by one de-compiles its round loop into a host loop — measured ~10x
    at m32. `testing/blake3_field_transcript_test.py::RoundLoopTest` is the
    leading indicator and needs no GPU.)

    Every op goes through `fs`, not through the transcript directly: an eager
    transcript op dispatches each of its internal primitives separately and
    re-traces the whole program on every call (measured ~1.4-2.1 s per
    `sample_f128` here against 0.1 ms through the cached hop). That is the
    `Sha256Challenger` arrangement, and routing around it is what made the
    first device implementation look 4.3x SLOWER than the host arm it replaced.
    """

    def __init__(self, domain: bytes):
        self._t = _initial_device_transcript(bytes(domain))

    def observe_label(self, label: bytes) -> None:
        self._t = fs.observe_label(self._t, label)

    def observe_bytes(self, data) -> None:
        if isinstance(data, (bytes, bytearray, memoryview)):
            data = np.frombuffer(data, np.uint8)
        else:
            data = fnp.asarray(data, fnp.uint8).reshape(-1)
        self._t = fs.observe_bytes(self._t, data)

    def observe_f128(self, g) -> None:
        if fnp.ndim(g) == 0:
            self._t = fs.observe_scalar(self._t, g)
        else:
            self._t = fs.observe_slice(self._t, g)

    def sample_f128(self, n: int | None = None):
        if n is None:
            self._t, g = fs.sample_scalar(self._t)
            return g
        self._t, g = fs.sample_slice(self._t, n)
        return g

    def grind_pow(self, bits: int) -> int:
        self._t, witness = fs.grind(self._t, bits)
        return int(witness)

    @property
    def field(self):
        return self._t.field

    @property
    def has_dedicated_fusion(self) -> bool:
        return self._t.has_dedicated_fusion

    def observe(self, values) -> "Blake3DeviceChallenger":
        self._t = self._t.observe(values)
        return self

    def sample(self, n: int = 1) -> tuple["Blake3DeviceChallenger", object]:
        self._t, out = self._t.sample(n)
        return self, out

    def observe_and_sample(
        self, values, n: int = 1
    ) -> tuple["Blake3DeviceChallenger", object]:
        self._t, out = self._t.observe_and_sample(values, n)
        return self, out

    def grind(self, pow_bits: int) -> tuple["Blake3DeviceChallenger", object]:
        self._t, witness = self._t.grind(pow_bits)
        return self, witness

    def check_witness(
        self, witness, *, pow_bits: int
    ) -> tuple["Blake3DeviceChallenger", object]:
        self._t, ok = self._t.check_witness(witness, pow_bits=pow_bits)
        return self, ok
