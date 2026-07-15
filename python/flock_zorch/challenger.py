"""flock's Fiat-Shamir challenger (SHA-256), authored over zorch's device
`Sha256FieldTranscript` — byte-identical to flock-core's `FsChallenger`.

The Merlin-over-SHA256 wire framing (op tags, u64-LE length prefixes,
`SHA256(buffer||ctr)` counter-squeeze, re-absorb, PoW) is zorch's streaming
device transcript instantiated with the native GHASH dtype, so observes and
samples carry F128 elements directly and the transcript state is a
scan-threadable pytree. Byte-identity across substrates is zorch's guarantee
(`sha256_field_transcript_test` pins it against `ByteHashTranscript`), so flock
keeps no gate of its own.

This wrapper keeps flock-core's `&mut self` API shape and the `uint64[..., 2]`
lane representation at its boundary; each `sample_*` therefore materializes the
challenge to host numpy. Sub-protocols migrate to threading the transcript
itself through their jitted rounds, at which point this wrapper shrinks away.

Requires `zorch` on PYTHONPATH (run gates with `PYTHONPATH=python:../zorch`).
"""
from __future__ import annotations

import numpy as np
import frx.numpy as jnp

from zorch.sha256_field_transcript import Sha256FieldTranscript

from flock_zorch import field, fs


class Challenger:
    """Mutable wrapper over the functional device transcript, mirroring flock's
    `&mut self` `FsChallenger` API. F128 values are `uint64[..., 2]` arrays at
    this boundary (the `field.py` representation); the transcript holds native
    `binary_field_ghash` elements."""

    def __init__(self, domain: bytes):
        self._t = Sha256FieldTranscript.new(domain, jnp.binary_field_ghash)

    def observe_label(self, label: bytes) -> None:
        self._t = fs.observe_label(self._t, label)

    def observe_bytes(self, data) -> None:
        self._t = fs.observe_bytes(
            self._t, np.frombuffer(bytes(data), np.uint8))

    def observe_f128(self, v) -> None:
        self._t = fs.observe_scalar(self._t, field.to_ghash(jnp.asarray(v)))

    def observe_f128_slice(self, vs) -> None:
        vs = jnp.asarray(np.asarray(vs, np.uint64).reshape(-1, 2))
        self._t = fs.observe_slice(self._t, field.to_ghash(vs))

    def sample_f128(self) -> np.ndarray:
        self._t, g = fs.sample_scalar(self._t)
        return field.from_ghash_host(g)

    def sample_f128_vec(self, n: int) -> np.ndarray:
        self._t, g = fs.sample_slice(self._t, n)
        return field.from_ghash_host(g)

    def grind_pow(self, bits: int) -> int:
        self._t, witness = fs.grind(self._t, bits)
        return int(witness)

    def verify_pow(self, nonce: int, bits: int) -> bool:
        self._t, ok = fs.check_witness(self._t, nonce, bits)
        return bool(np.asarray(ok))
