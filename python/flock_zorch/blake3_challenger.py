# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BLAKE3 Fiat-Shamir challenger — the flock-challenge benchmark profile's FS arm.

The ranked benchmark (Layr-Labs/flock-challenge) pins its transcript hash to
BLAKE3: the worker and the harness verifier both run
`FsChallenger::with_hash(domain, Blake3)`, so a proof aimed at their window has
to draw every challenge from this arm. Its framing is the same Merlin-over-hash
duplex the SHA-256 challenger speaks — identical op tags, length prefixes, and
squeeze re-absorb, i.e. zorch's `ByteHashTranscript` wire — with two per-hash
divergences: the squeeze is an XOF read rather than a counter chain, and the PoW
pre-image is one whole 64-byte block rather than 40.
`testing/_blake3_challenger.py` spells both out and carries the host rows that
gate this one.

`Blake3DeviceChallenger` is the prove path: `Sha256Challenger`'s surface over
the device `Blake3FieldTranscript`, whose state is a fixed-shape pytree, so the
four transcript-threading jitted zones (zerocheck ML ladder, lincheck
prove_inf_product, ring-switch reduce, the Ligerito open) carry it through their
loops instead of de-compiling into host loops.

Byte-gated against transcripts dumped from the fork (d866043) by
`testing/blake3_challenger_test.py`, which also carries the dump recipe;
`testing/blake3_field_transcript_test.py` pins this row against the host one,
op for op.
"""

from __future__ import annotations

import functools

import frx.numpy as fnp
import numpy as np
from zorch.blake3_field_transcript import Blake3FieldTranscript

from flock_zorch import fs

# The fork's BLAKE3 PoW pre-image length: state digest (32) ‖ nonce (8) ‖ zero
# padding to one whole block. The host rows under `testing/` read it from here
# so the two arms cannot disagree on the wire.
_POW_BLOCK = 64


@functools.lru_cache(maxsize=None)
def _initial_device_transcript(domain: bytes):
    """Memoize the seeded device state per domain.

    `Blake3FieldTranscript.new` absorbs the domain through the full device
    absorb program — 0.5 s of trace per construction. Array values are
    immutable and every challenger replaces `_t` rather than mutating it, so
    one seeded state is safe to share between proves. `Sha256Challenger` does
    the same for the same reason.

    `pow_preimage_bytes` is the fork's, not zorch's default: the fork pads its
    PoW pre-image to a whole block where the canonical wire stops at 40. Drop
    the argument and everything still compiles and most tests still pass — only
    the proof goldens catch it, because every challenge after the grind moves.
    """
    return Blake3FieldTranscript.new(
        domain, fnp.binary_field_ghash, pow_preimage_bytes=_POW_BLOCK
    )


class Blake3DeviceChallenger:
    """`Sha256Challenger`'s surface over the device transcript — the BLAKE3
    profile's prove-path challenger, and the structural twin of the SHA-256 one.

    `Blake3FieldTranscript`'s state is a fixed-shape pytree, so a jitted round
    loop carries it and the sumcheck loop stays inside the compiled program.
    (The host row under `testing/_blake3_challenger.py` cannot: a host
    transcript is not a pytree, so a prove driven by one de-compiles its round
    loop back onto the host.
    `testing/blake3_field_transcript_test.py::RoundLoopTest` is the leading
    indicator and needs no GPU.)

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
