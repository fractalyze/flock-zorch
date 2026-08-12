# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The fork's BLAKE3 Fiat-Shamir transcript, on device and jit-threadable.

`Sha256FieldTranscript`'s role for the benchmark profile's hash arm: the Merlin
framing over a resumable state, so observes and samples carry native
`binary_field_ghash` elements and the transcript itself is a pytree a jitted
round loop can carry.

**Why it exists.** `Blake3CallbackChallenger` keeps its state on the host behind
`io_callback`. That is correct but un-threadable, and the cost is not the host
hop: with it in place the sumcheck's round loop cannot compile into the prove
program at all (`jit__mlv_sumcheck` drops from 35,992 HLO instructions with 192
`while` loops to 2,912 with none), which is ~10x of the window snark.fast
scores. The callback arm stays as the byte oracle.

The wire is `zorch.byte_transcript`'s, with the fork's two BLAKE3 deviations:
the squeeze is an XOF read rather than a counter-block squeeze, and the PoW
pre-image is padded to a whole 64-byte block. `Blake3ByteTranscript` pins both
on the host; `blake3_field_transcript_test` pins this against it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import frx.numpy as fnp
import numpy as np
from frx import lax
from frx.tree_util import register_dataclass
from hash_frx.blake3 import blake3
from hash_frx.blake3.compress import CHUNK_END, CHUNK_START, ROOT, compress
from zorch.byte_transcript import (
    KIND_SCALAR,
    KIND_SLICE,
    OP_BYTES,
    OP_DOMAIN,
    OP_LABEL,
    OP_OBSERVE,
    OP_SQUEEZE,
    _len8,
    _validate_pow_bits,
)

from flock_zorch.hash import blake3_stream

U32 = fnp.uint32
_DIGEST_BYTES = 32
# The fork's PoW pre-image: digest (32) ‖ nonce (8) ‖ zero padding to one block.
_POW_BLOCK = 64
# Nonces tested per batched compression. A hit lands in the first window for any
# practical `bits`, so this is a batching knob and not a bound: windows are
# scanned in ascending order and the first hit inside one is that window's
# smallest, so the answer is the globally smallest nonce at any window size.
_GRIND_WINDOW = 1 << 12


def _const_u8(data: bytes):
    return fnp.asarray(np.frombuffer(data, np.uint8), fnp.uint8)


def _le_words(u8):
    """uint8 `[..., 4k]` -> uint32 `[..., k]`, little-endian."""
    b = u8.reshape(*u8.shape[:-1], u8.shape[-1] // 4, 4).astype(U32)
    return (
        b[..., 0]
        | (b[..., 1] << U32(8))
        | (b[..., 2] << U32(16))
        | (b[..., 3] << U32(24))
    )


def _le_bytes(words):
    """uint32 `[..., k]` -> uint8 `[..., 4k]`, little-endian (inverse)."""
    shifts = fnp.asarray([0, 8, 16, 24], U32)
    b = ((words[..., None] >> shifts) & U32(0xFF)).astype(fnp.uint8)
    return b.reshape(*words.shape[:-1], words.shape[-1] * 4)


@partial(register_dataclass, data_fields=["state"], meta_fields=["dtype"])
@dataclass(frozen=True)
class Blake3FieldTranscript:
    """Merlin-over-BLAKE3 on a resumable device state.

    `state` is the only data field, so the pytree structure is fixed no matter
    how much has been absorbed — the property that lets a round loop carry it.
    """

    state: blake3_stream.Blake3Stream
    dtype: Any

    @property
    def field(self):
        return self.dtype

    @property
    def has_dedicated_fusion(self) -> bool:
        # No fused-FS marker exists for the BLAKE3 arm, so consumers take their
        # plain decomposition paths. Measured not to matter for the prove's
        # shape: forcing this flag on the SHA-256 arm left the compiled program
        # byte-for-byte identical.
        return False

    @classmethod
    def new(cls, domain: bytes, dtype: Any) -> Blake3FieldTranscript:
        seed = _const_u8(bytes([OP_DOMAIN]) + _len8(len(domain)) + bytes(domain))
        return cls(blake3_stream.absorb(blake3_stream.init(), seed), np.dtype(dtype))

    # ---- absorb ------------------------------------------------------------
    def _item_bytes(self) -> int:
        return int(np.dtype(self.dtype).itemsize)

    def _absorb(self, payload) -> Blake3FieldTranscript:
        return replace(self, state=blake3_stream.absorb(self.state, payload))

    def _elem_bytes(self, values):
        return lax.bitcast_convert_type(values, fnp.uint8)

    def _u8_to_elems(self, u8, n: int):
        return lax.bitcast_convert_type(
            u8.reshape(n, self._item_bytes()), self.dtype
        ).reshape(n)

    def observe(self, values) -> Blake3FieldTranscript:
        """Slice framing: `[OP_OBSERVE, KIND_SLICE] || len8(count) || bytes`."""
        vals_u8 = self._elem_bytes(values).reshape(-1)
        count = int(vals_u8.shape[0]) // self._item_bytes()
        framing = _const_u8(bytes([OP_OBSERVE, KIND_SLICE]) + _len8(count))
        return self._absorb(fnp.concatenate([framing, vals_u8]))

    def observe_scalar(self, value) -> Blake3FieldTranscript:
        """Scalar framing `[OP_OBSERVE, KIND_SCALAR] || bytes`, no length — a
        scalar's width is implicit in the dtype. A 0-d value is one op; `[n]` is
        n ops in order, built as one absorb payload."""
        vals_u8 = self._elem_bytes(value).reshape(-1, self._item_bytes())
        framing = fnp.broadcast_to(
            _const_u8(bytes([OP_OBSERVE, KIND_SCALAR])), (vals_u8.shape[0], 2)
        )
        return self._absorb(fnp.concatenate([framing, vals_u8], axis=1).reshape(-1))

    def observe_label(self, label: bytes) -> Blake3FieldTranscript:
        return self._absorb(
            _const_u8(bytes([OP_LABEL]) + _len8(len(label)) + bytes(label))
        )

    def observe_bytes(self, data) -> Blake3FieldTranscript:
        data = fnp.asarray(data, fnp.uint8).reshape(-1)
        framing = _const_u8(bytes([OP_BYTES]) + _len8(int(data.shape[0])))
        return self._absorb(fnp.concatenate([framing, data]))

    # ---- squeeze -----------------------------------------------------------
    def _squeeze(self, framing, nbytes: int):
        """Absorb `framing`, read `nbytes` of extendable output, re-absorb it.

        The XOF read is the fork's deviation from the base transcript's
        counter-block squeeze: BLAKE3's output IS extendable, so one read of
        length `nbytes` is the whole squeeze.
        """
        t = self._absorb(framing)
        out = blake3_stream.finalize(t.state, nbytes)
        return t._absorb(out), out

    def sample(self, n: int = 1) -> tuple[Blake3FieldTranscript, Any]:
        framing = _const_u8(bytes([OP_SQUEEZE, KIND_SLICE]) + _len8(n))
        t, out = self._squeeze(framing, n * self._item_bytes())
        return t, t._u8_to_elems(out, n)

    def sample_scalar(self) -> tuple[Blake3FieldTranscript, Any]:
        """Distinct wire from `sample(1)` — the KIND tag differs."""
        framing = _const_u8(bytes([OP_SQUEEZE, KIND_SCALAR]))
        t, out = self._squeeze(framing, self._item_bytes())
        return t, t._u8_to_elems(out, 1)[0]

    def observe_and_sample(self, values, n: int = 1):
        return self.observe(values).sample(n)

    # ---- proof of work -----------------------------------------------------
    def _pow_digests(self, digest_u8, base):
        """Digests of `_GRIND_WINDOW` PoW pre-images from `base`: uint8
        `[W, 32]`.

        A pre-image is exactly one 64-byte block, so its BLAKE3 is a single
        compression — no chunk or tree machinery, just the three flags a lone
        block carries.
        """
        w = _GRIND_WINDOW
        nonces = base + fnp.arange(w, dtype=fnp.uint64)
        block = fnp.zeros((w, 16), U32)
        block = block.at[:, :8].set(fnp.broadcast_to(_le_words(digest_u8), (w, 8)))
        block = block.at[:, 8].set((nonces & fnp.uint64(0xFFFF_FFFF)).astype(U32))
        block = block.at[:, 9].set((nonces >> fnp.uint64(32)).astype(U32))
        out = compress(
            fnp.broadcast_to(blake3.hash_mode().key_words, (w, 8)),
            block,
            fnp.zeros((w, 2), U32),
            fnp.full((w,), _POW_BLOCK, U32),
            fnp.full(
                (w,), blake3.hash_mode().flags | CHUNK_START | CHUNK_END | ROOT, U32
            ),
        )
        return _le_bytes(out[:, :8])

    def _hits(self, digests, bits: int):
        """Which digests have `bits` leading zero bits: bool `[W]`."""
        full, rem = divmod(bits, 8)
        ok = fnp.all(digests[:, :full] == fnp.uint8(0), axis=1)
        if rem:
            ok = ok & ((digests[:, full] >> fnp.uint8(8 - rem)) == fnp.uint8(0))
        return ok

    def grind(self, pow_bits: int) -> tuple[Blake3FieldTranscript, Any]:
        """Lowest u64 nonce whose PoW passes, then absorb it so later challenges
        bind to it. `bits == 0` is nonce 0 without a search, as on the host."""
        _validate_pow_bits(pow_bits, _DIGEST_BYTES)
        if pow_bits == 0:
            nonce = fnp.uint64(0)
        else:
            digest = blake3_stream.finalize(self.state, _DIGEST_BYTES)

            def cond(carry):
                base, found = carry
                del base
                return ~found

            def body(carry):
                base, _ = carry
                hits = self._hits(self._pow_digests(digest, base), pow_bits)
                any_hit = fnp.any(hits)
                # `argmax` of a bool picks the first True, which is the smallest
                # nonce in this window; ascending windows make it the global one.
                first = fnp.argmax(hits).astype(fnp.uint64)
                return fnp.where(any_hit, base + first, base + _GRIND_WINDOW), any_hit

            nonce, _ = lax.while_loop(cond, body, (fnp.uint64(0), fnp.bool_(False)))
        return self.observe_bytes(_le_bytes(_nonce_words(nonce))), nonce

    def check_witness(self, witness, *, pow_bits: int):
        """Verifier mirror: check, then absorb the nonce REGARDLESS so the two
        transcripts stay in lockstep."""
        _validate_pow_bits(pow_bits, _DIGEST_BYTES)
        nonce = fnp.asarray(witness, fnp.uint64).reshape(())
        if pow_bits == 0:
            ok = nonce == fnp.uint64(0)
        else:
            digest = blake3_stream.finalize(self.state, _DIGEST_BYTES)
            one = _one_pow_digest(digest, nonce)
            ok = self._hits(one[None, :], pow_bits)[0]
        return self.observe_bytes(_le_bytes(_nonce_words(nonce))), ok


def _nonce_words(nonce):
    """u64 nonce -> uint32 [2], little-endian — the transcript's only integer
    encoding."""
    return fnp.stack(
        [
            (nonce & fnp.uint64(0xFFFF_FFFF)).astype(U32),
            (nonce >> fnp.uint64(32)).astype(U32),
        ]
    )


def _one_pow_digest(digest_u8, nonce):
    block = fnp.zeros((1, 16), U32)
    block = block.at[0, :8].set(_le_words(digest_u8))
    block = block.at[0, 8:10].set(_nonce_words(nonce))
    out = compress(
        blake3.hash_mode().key_words[None, :],
        block,
        fnp.zeros((1, 2), U32),
        fnp.full((1,), _POW_BLOCK, U32),
        fnp.full((1,), blake3.hash_mode().flags | CHUNK_START | CHUNK_END | ROOT, U32),
    )
    return _le_bytes(out[:, :8])[0]
