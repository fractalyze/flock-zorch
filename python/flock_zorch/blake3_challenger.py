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
case the host rows exist for. Two consumers sit on the byte transcript:

- `Blake3Challenger` — the eager reference (`Sha256Challenger`'s flock-core
  API over host bytes), the twin everything else is pinned against.
- `Blake3CallbackTranscript` / `Blake3CallbackChallenger` — the prove-path
  arm: the device transcript's method surface as a pytree whose every op is
  one ordered `io_callback` into the host transcript, so the four
  transcript-threading jitted zones (zerocheck ML ladder, lincheck
  prove_inf_product, ring-switch reduce, the Ligerito open) run UNCHANGED
  under the benchmark profile.

Byte-gated against transcripts dumped from the fork (d866043) by
`testing/blake3_challenger_test.py`, which also carries the dump recipe.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.experimental import io_callback
from frx.tree_util import register_dataclass
from hash_frx.blake3.byte_hashes import HostBlake3
from zorch.byte_transcript import (
    ByteHashTranscript,
    _leading_zero_bits_ok,
    _len8,
    _validate_pow_bits,
)

from flock_zorch.ghash import _ghash_to_lanes, _lanes_to_ghash, from_ghash, to_ghash

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


# --- jit-threadable arm: the callback transcript -----------------------------
#
# The prove path threads its transcript as a traced pytree through jitted
# programs (the zerocheck ML ladder, lincheck's prove_inf_product, the whole
# Ligerito open). `Blake3ByteTranscript` is a host object and cannot ride
# those, and a device BLAKE3 duplex does not exist (BLAKE3 is a tree/XOF, not
# a fixed-midstate Merkle-Damgard stream like SHA-256). The bridge: a pytree
# with the device transcript's method surface whose every op is one ORDERED
# `io_callback` into a host `Blake3ByteTranscript`. The jitted programs run
# unchanged; the FS chain — which is serial by construction — runs on the
# host, exactly where the fork itself runs it.
#
# The pytree carries two uint32 scalars and nothing static: `token` threads a
# data dependence through every op (belt to the ordered-effects braces), and
# `handle` keys the host-state registry AS A TRACED VALUE — a static handle
# would make every new transcript a new jit cache key, recompiling the prove
# per trial. Field elements cross the callback boundary as uint64 lanes
# (zk_dtypes do not ride io_callback); the bitcast lives at the edges.

_CALLBACK_STATES: dict[int, Blake3ByteTranscript] = {}
_HANDLES = itertools.count()


def _state(handle) -> Blake3ByteTranscript:
    return _CALLBACK_STATES[int(handle)]


# Module-level callbacks (stable identity across traces); the label/count
# arguments ride as arrays so nothing is closure-captured.


def _cb_observe_label(handle, tok, label_u8):
    _CALLBACK_STATES[int(handle)] = _state(handle).observe_label(
        np.asarray(label_u8).tobytes()
    )
    return np.asarray(tok)


def _cb_observe_bytes(handle, tok, data_u8):
    _CALLBACK_STATES[int(handle)] = _state(handle).observe_bytes(
        np.asarray(data_u8).tobytes()
    )
    return np.asarray(tok)


def _cb_observe_scalar(handle, tok, lanes):
    t = _state(handle)
    for row in np.asarray(lanes).reshape(-1, 2):
        t = t.observe_scalar(np.ascontiguousarray(row, dtype=np.uint64).tobytes())
    _CALLBACK_STATES[int(handle)] = t
    return np.asarray(tok)


def _cb_observe_slice(handle, tok, lanes):
    rows = np.ascontiguousarray(np.asarray(lanes).reshape(-1, 2), dtype=np.uint64)
    _CALLBACK_STATES[int(handle)] = _state(handle).observe_slice(
        rows.tobytes(), rows.shape[0]
    )
    return np.asarray(tok)


def _cb_sample_scalar(handle, tok):
    t, buf = _state(handle).sample_scalar(_F128_BYTES)
    _CALLBACK_STATES[int(handle)] = t
    return np.asarray(tok), np.frombuffer(buf, np.uint64).copy()


def _cb_sample_slice(handle, tok, n):
    n = int(n)
    t, buf = _state(handle).sample_slice(n, _F128_BYTES)
    _CALLBACK_STATES[int(handle)] = t
    return np.asarray(tok), np.frombuffer(buf, np.uint64).reshape(n, 2).copy()


def _cb_grind(handle, tok, bits):
    t, nonce = _state(handle).grind_pow(int(bits))
    _CALLBACK_STATES[int(handle)] = t
    return np.asarray(tok), np.uint64(nonce)


def _cb_check_witness(handle, tok, witness, bits):
    t, ok = _state(handle).verify_pow(int(witness), bits=int(bits))
    _CALLBACK_STATES[int(handle)] = t
    return np.asarray(tok), np.bool_(ok)


_U32 = fnp.uint32
_U64 = fnp.uint64


def _tok_shape():
    return frx.ShapeDtypeStruct((), _U32)


@register_dataclass
@dataclasses.dataclass(frozen=True)
class Blake3CallbackTranscript:
    """The device transcript's method surface over a host BLAKE3 byte
    transcript, one ordered `io_callback` per op. Threads jitted zones and
    `lax.scan` bodies unchanged; byte-gated (eager AND through one jitted
    program) against the fork fixtures by `testing/blake3_challenger_test.py`.

    `field`/`has_dedicated_fusion` mirror `Sha256FieldTranscript`'s surface;
    `has_dedicated_fusion` is False, so consumers take their plain
    decomposition paths (no fused-FS marker exists for this arm)."""

    token: Array
    handle: Array

    @classmethod
    def new(cls, domain: bytes) -> Blake3CallbackTranscript:
        handle = next(_HANDLES)
        _CALLBACK_STATES[handle] = Blake3ByteTranscript.new(bytes(domain), HostBlake3())
        return cls(token=fnp.zeros((), _U32), handle=fnp.full((), handle, dtype=_U32))

    @property
    def field(self):
        return fnp.binary_field_ghash

    @property
    def has_dedicated_fusion(self) -> bool:
        return False

    def _tok(self, cb, *args) -> Blake3CallbackTranscript:
        tok = io_callback(
            cb, _tok_shape(), self.handle, self.token, *args, ordered=True
        )
        return dataclasses.replace(self, token=tok)

    def observe_label(self, label: bytes) -> Blake3CallbackTranscript:
        label_u8 = np.frombuffer(bytes(label), np.uint8)
        return self._tok(_cb_observe_label, label_u8)

    def observe_bytes(self, data) -> Blake3CallbackTranscript:
        return self._tok(_cb_observe_bytes, fnp.asarray(data, fnp.uint8).reshape(-1))

    def observe_scalar(self, x) -> Blake3CallbackTranscript:
        """Scalar-framed observe — `x` 0-d for one op, `[n]` for n ops (the
        `Sha256FieldTranscript.observe_scalar` contract)."""
        return self._tok(_cb_observe_scalar, from_ghash(x).reshape(-1, 2))

    def observe(self, values) -> Blake3CallbackTranscript:
        return self._tok(_cb_observe_slice, from_ghash(values).reshape(-1, 2))

    def sample_scalar(self) -> tuple[Blake3CallbackTranscript, Array]:
        tok, lanes = io_callback(
            _cb_sample_scalar,
            (_tok_shape(), frx.ShapeDtypeStruct((2,), _U64)),
            self.handle,
            self.token,
            ordered=True,
        )
        return dataclasses.replace(self, token=tok), to_ghash(lanes)

    def sample(self, n: int) -> tuple[Blake3CallbackTranscript, Array]:
        tok, lanes = io_callback(
            _cb_sample_slice,
            (_tok_shape(), frx.ShapeDtypeStruct((n, 2), _U64)),
            self.handle,
            self.token,
            np.uint32(n),
            ordered=True,
        )
        return dataclasses.replace(self, token=tok), to_ghash(lanes)

    def observe_and_sample(self, values, n: int = 1):
        return self.observe(values).sample(n)

    def grind(self, pow_bits: int) -> tuple[Blake3CallbackTranscript, Array]:
        tok, nonce = io_callback(
            _cb_grind,
            (_tok_shape(), frx.ShapeDtypeStruct((), _U64)),
            self.handle,
            self.token,
            np.uint32(pow_bits),
            ordered=True,
        )
        return dataclasses.replace(self, token=tok), nonce

    def check_witness(self, witness, *, pow_bits: int):
        tok, ok = io_callback(
            _cb_check_witness,
            (_tok_shape(), frx.ShapeDtypeStruct((), fnp.bool_)),
            self.handle,
            self.token,
            fnp.asarray(witness, _U64),
            np.uint32(pow_bits),
            ordered=True,
        )
        return dataclasses.replace(self, token=tok), ok


class Blake3CallbackChallenger:
    """`Sha256Challenger`'s exact surface over the callback transcript — the
    benchmark profile's prove-path challenger. Eager touchpoints call the
    wrapper methods; the four transcript-threading zones take and reassign
    `._t` (the `Blake3CallbackTranscript` pytree) exactly as they do the
    device SHA-256 transcript."""

    def __init__(self, domain: bytes):
        self._t = Blake3CallbackTranscript.new(bytes(domain))

    def observe_label(self, label: bytes) -> None:
        self._t = self._t.observe_label(label)

    def observe_bytes(self, data) -> None:
        if isinstance(data, (bytes, bytearray, memoryview)):
            data = np.frombuffer(data, np.uint8)
        self._t = self._t.observe_bytes(data)

    def observe_f128(self, g) -> None:
        """Observe F128 — a scalar or a slice, framed by shape (flock
        scalar-frames a single element, slice-frames many)."""
        if fnp.ndim(g) == 0:
            self._t = self._t.observe_scalar(g)
        else:
            self._t = self._t.observe(g)

    def sample_f128(self, n: int | None = None):
        """Bare `sample_f128()` is a scalar draw; `sample_f128(n)` a length-`n`
        slice — the two frame differently on the wire."""
        if n is None:
            self._t, g = self._t.sample_scalar()
            return g
        self._t, g = self._t.sample(n)
        return g

    def grind_pow(self, bits: int) -> int:
        self._t, witness = self._t.grind(bits)
        return int(witness)

    # --- zorch `Transcript` seam (mirrors `Sha256Challenger`) ---------------

    @property
    def field(self):
        return self._t.field

    @property
    def has_dedicated_fusion(self) -> bool:
        return self._t.has_dedicated_fusion

    def observe(self, values) -> "Blake3CallbackChallenger":
        self._t = self._t.observe(values)
        return self

    def sample(self, n: int = 1) -> tuple["Blake3CallbackChallenger", object]:
        self._t, out = self._t.sample(n)
        return self, out

    def observe_and_sample(
        self, values, n: int = 1
    ) -> tuple["Blake3CallbackChallenger", object]:
        self._t, out = self._t.observe_and_sample(values, n)
        return self, out

    def grind(self, pow_bits: int) -> tuple["Blake3CallbackChallenger", object]:
        self._t, witness = self._t.grind(pow_bits)
        return self, witness

    def check_witness(
        self, witness, *, pow_bits: int
    ) -> tuple["Blake3CallbackChallenger", object]:
        self._t, ok = self._t.check_witness(witness, pow_bits=pow_bits)
        return self, ok
