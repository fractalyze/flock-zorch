"""Flock's immutable Fiat–Shamir transcript.

The wire format is byte-identical to flock-core's SHA-256 ``FsChallenger``.
Unlike Rust's ``&mut self`` surface, Python protocol code threads the returned
transcript explicitly. This makes transcript state ordinary stage data and
prevents a prover or verifier from advancing hidden shared state.

Flock's wire format distinguishes a scalar F128 operation from a length-one
F128 slice operation. The explicit ``observe_f128`` / ``sample_f128`` methods
preserve that distinction. The generic zorch ``Transcript`` methods implement
the framing required by the Ligerito choreography.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array
from frx.tree_util import register_dataclass

from zorch.sha256_field_transcript import Sha256FieldTranscript

from flock_zorch import fs


@functools.partial(register_dataclass, data_fields=["inner"], meta_fields=[])
@dataclass(frozen=True)
class FlockTranscript:
    """Immutable, device-threadable transcript with Flock wire framing."""

    inner: Sha256FieldTranscript

    @classmethod
    def new(cls, domain: bytes) -> "FlockTranscript":
        return cls(Sha256FieldTranscript.new(domain, fnp.binary_field_ghash))

    @property
    def has_dedicated_fusion(self) -> bool:
        return self.inner.has_dedicated_fusion

    # Flock protocol operations. A scalar and a one-element slice are
    # deliberately different operations on the wire.
    def observe_label(self, label: bytes) -> "FlockTranscript":
        return FlockTranscript(fs.observe_label(self.inner, label))

    def observe_bytes(self, data) -> "FlockTranscript":
        values = (
            np.frombuffer(bytes(data), np.uint8)
            if isinstance(data, (bytes, bytearray, memoryview))
            else fnp.asarray(data, fnp.uint8)
        )
        return FlockTranscript(fs.observe_bytes(self.inner, values))

    def observe_f128(self, values) -> "FlockTranscript":
        values = fnp.asarray(values)
        if fnp.ndim(values) == 0:
            return FlockTranscript(fs.observe_scalar(self.inner, values))
        return FlockTranscript(fs.observe_slice(self.inner, values))

    def sample_f128(self, n: int | None = None) -> tuple["FlockTranscript", Array]:
        if n is None:
            inner, value = fs.sample_scalar(self.inner)
        else:
            inner, value = fs.sample_slice(self.inner, n)
        return FlockTranscript(inner), value

    def grind_pow(self, bits: int) -> tuple["FlockTranscript", Array]:
        inner, witness = fs.grind(self.inner, bits)
        return FlockTranscript(inner), witness

    def verify_pow(
        self, nonce: int | Array, bits: int
    ) -> tuple["FlockTranscript", Array]:
        inner, ok = fs.check_witness(self.inner, nonce, bits)
        return FlockTranscript(inner), ok

    # zorch Transcript protocol. Ligerito observes field messages as a sequence
    # of scalar operations and treats sample(1) as one scalar squeeze.
    def observe(self, values: Array) -> "FlockTranscript":
        values = fnp.asarray(values)
        if values.dtype == fnp.uint8:
            return FlockTranscript(fs.observe_bytes(self.inner, values))
        if values.dtype != fnp.binary_field_ghash:
            raise TypeError(f"no Flock framing for observed dtype {values.dtype}")
        return FlockTranscript(fs.observe_scalar(self.inner, values.reshape(-1)))

    def sample(self, n: int = 1) -> tuple["FlockTranscript", Array]:
        if n == 1:
            inner, value = fs.sample_scalar(self.inner)
            return FlockTranscript(inner), value.reshape(1)
        inner, values = fs.sample_slice(self.inner, n)
        return FlockTranscript(inner), values

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple["FlockTranscript", Array]:
        return self.observe(values).sample(n)


def flock_transcript(domain: bytes) -> FlockTranscript:
    """Construct a fresh Flock transcript."""
    return FlockTranscript.new(domain)
