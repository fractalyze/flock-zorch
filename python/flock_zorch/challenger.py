"""flock's Fiat-Shamir challenger (SHA-256), authored over zorch's generic
byte-duplex transcript — byte-identical to flock-core's `FsChallenger`.

The Merlin-over-SHA256 wire framing (op tags, u64-LE length prefixes,
`SHA256(buffer||ctr)` counter-squeeze, re-absorb, PoW) is the scheme-agnostic
`zorch.byte_transcript.ByteHashTranscript`, parameterized by an injected
`ByteHash` (`HashlibSha256()` host hashlib / `Sha256()` device marker). This
module is the thin flock glue: it serializes an F128 = uint64[2] lane pair as 16
bytes (`lo_le8 || hi_le8`) and reinterprets squeezed bytes back. Host-side /
sequential, matching flock's non-negotiable #3 (Fiat-Shamir runs on the host;
bulk arithmetic on device).

Requires `zorch` on PYTHONPATH (run gates with `PYTHONPATH=python:../zorch`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zorch.byte_transcript import ByteHashTranscript
from zorch.hash.sha256 import HashlibSha256

if TYPE_CHECKING:
    from zorch.hash.byte_hash import ByteHash


def _f128_bytes(v) -> bytes:
    """F128 (uint64[2] = [lo, hi]) -> 16 LE bytes = lo.to_le8 || hi.to_le8."""
    return np.asarray(v, dtype="<u8").reshape(2).tobytes()


def _f128s_bytes(vs) -> bytes:
    """[n, 2] F128 -> n*16 bytes, each element lo_le8 || hi_le8, in order."""
    return np.ascontiguousarray(np.asarray(vs, dtype="<u8").reshape(-1, 2)).tobytes()


def _f128_from(buf: bytes) -> np.ndarray:
    return np.frombuffer(buf, dtype="<u8").reshape(2).astype(np.uint64)


def _f128s_from(buf: bytes, n: int) -> np.ndarray:
    return np.frombuffer(buf, dtype="<u8").reshape(n, 2).astype(np.uint64)


class Challenger:
    """Mutable wrapper over a functional byte transcript, mirroring flock's
    `&mut self` `FsChallenger` API. F128 values are `uint64[..., 2]` arrays (the
    `field.py` representation).

    `byte_hash` selects the backend injected into the one
    `zorch.byte_transcript.ByteHashTranscript`: `None` (the default) uses the host
    `HashlibSha256` (hashlib), or pass the byte-identical device `Sha256` (SHA-256
    on the `zorch.sha256` marker), which the `challenger_device_oracle_test` gate
    pins to the same flock golden. Both expose the same byte-framed transcript, so
    the glue here is unchanged."""

    def __init__(self, domain: bytes, *, byte_hash: ByteHash | None = None):
        if byte_hash is None:
            byte_hash = HashlibSha256()
        self._t = ByteHashTranscript.new(domain, byte_hash)

    def observe_label(self, label: bytes) -> None:
        self._t = self._t.observe_label(label)

    def observe_bytes(self, data: bytes) -> None:
        self._t = self._t.observe_bytes(bytes(data))

    def observe_f128(self, v) -> None:
        self._t = self._t.observe_scalar(_f128_bytes(v))

    def observe_f128_slice(self, vs) -> None:
        vs = np.asarray(vs, dtype=np.uint64).reshape(-1, 2)
        self._t = self._t.observe_slice(_f128s_bytes(vs), int(vs.shape[0]))

    def sample_f128(self) -> np.ndarray:
        self._t, buf = self._t.sample_scalar(16)
        return _f128_from(buf)

    def sample_f128_vec(self, n: int) -> np.ndarray:
        self._t, buf = self._t.sample_slice(n, 16)
        return _f128s_from(buf, n)

    def grind_pow(self, bits: int) -> int:
        self._t, nonce = self._t.grind_pow(bits)
        return nonce

    def verify_pow(self, nonce: int, bits: int) -> bool:
        self._t, ok = self._t.verify_pow(nonce, bits)
        return ok
