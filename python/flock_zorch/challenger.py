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

import frx.numpy as fnp
import numpy as np
from zorch.sha256_field_transcript import Sha256FieldTranscript

from flock_zorch import fs


class Challenger:
    """Mutable wrapper over the functional device transcript, mirroring flock's
    `&mut self` `FsChallenger` API. Observes and samples carry native
    `binary_field_ghash` elements; host-int consumers convert at their own edge."""

    def __init__(self, domain: bytes):
        self._t = Sha256FieldTranscript.new(domain, fnp.binary_field_ghash)

    def observe_label(self, label: bytes) -> None:
        self._t = fs.observe_label(self._t, label)

    def observe_bytes(self, data) -> None:
        self._t = fs.observe_bytes(self._t, np.frombuffer(bytes(data), np.uint8))

    def observe_f128(self, g) -> None:
        """Observe F128 (native `binary_field_ghash`) — a scalar or a slice,
        framed by shape (flock scalar-frames a single element, slice-frames many)."""
        if fnp.ndim(g) == 0:
            self._t = fs.observe_scalar(self._t, g)
        else:
            self._t = fs.observe_slice(self._t, g)

    def sample_f128(self, n: int | None = None):
        """Sample F128 as native `binary_field_ghash`. Bare `sample_f128()` is a
        single scalar draw; `sample_f128(n)` is a length-`n` slice — the two frame
        differently on the wire, so a length-1 vector still passes an explicit `n=1`
        (scalar vs slice(1) are NOT the same bytes)."""
        if n is None:
            self._t, g = fs.sample_scalar(self._t)
            return g
        self._t, g = fs.sample_slice(self._t, n)
        return g

    def grind_pow(self, bits: int) -> int:
        self._t, witness = fs.grind(self._t, bits)
        return int(witness)

    # --- zorch `Transcript` seam ---------------------------------------------
    #
    # The wrapper already *is* a transcript; these name the surface zorch's
    # composition roles are typed on, so a flock round can declare the seam it
    # threads instead of being unrepresentable to the type checker. Each one
    # delegates to the wrapped functional transcript and returns `self`, which
    # is the wrapper's existing `&mut self` contract — callers already rely on
    # mutation rather than on the returned value being a distinct object.

    @property
    def field(self):
        return self._t.field

    @property
    def has_dedicated_fusion(self) -> bool:
        return self._t.has_dedicated_fusion

    def observe(self, values) -> "Challenger":
        self._t = self._t.observe(values)
        return self

    def sample(self, n: int = 1) -> tuple["Challenger", object]:
        self._t, out = self._t.sample(n)
        return self, out

    def observe_and_sample(self, values, n: int = 1) -> tuple["Challenger", object]:
        self._t, out = self._t.observe_and_sample(values, n)
        return self, out

    def grind(self, pow_bits: int) -> tuple["Challenger", object]:
        self._t, witness = self._t.grind(pow_bits)
        return self, witness

    def check_witness(self, witness, *, pow_bits: int) -> tuple["Challenger", object]:
        self._t, ok = self._t.check_witness(witness, pow_bits=pow_bits)
        return self, ok
