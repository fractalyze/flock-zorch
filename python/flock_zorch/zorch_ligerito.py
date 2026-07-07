"""flock instantiation of zorch's Ligerito Fiat-Shamir seams (flock-zorch#32 T4).

zorch's `zorch.pcs.ligerito` routes every transcript interaction through the
`LigeritoChoreography` seam over a generic `Transcript`. This module supplies the
flock side of both seams so the code-generic driver produces flock's byte wire:

- `FlockTranscript`: zorch's functional `Transcript` protocol over the byte
  challenger substrate (`ByteHashTranscript`, the same one `challenger.Challenger`
  wraps). Framing is flock's: an F128 observe is a 16-byte `observe_scalar`
  (lo‖hi LE) per element, a root a raw `observe_bytes`, one challenge a
  `sample_scalar(16)` and an n-vector one `sample_slice(n, 16)`.
- `FlockChoreography`: flock `pcs::ligerito`'s FS shape — the
  `flock-ligerito-basis-v0` statement binding (claim + root, no point), eager
  message emission, tapered per-fold PoW (>0-conditional), unconditional
  per-level query PoW (0 bits still puts a nonce on the wire), and flock's
  rejection-sampled distinct sorted queries — with the byte `grind_pow` /
  `verify_pow` as the grind mechanism.
- `flock_ligerito_config`: the `dump_ligerito` config block as a zorch
  `LigeritoConfig` + `FlockChoreography` pair (LSB-first alpha weights,
  compressed `(c0, c2)` round messages).

The algebra side rides zorch's `LigeritoConfig.monomial_commit` (flock commits
the raw lanes — the bit-reversed coefficient basis) plus the raw-basis entries
(`open_with_basis` / `verify_with_basis`): the driver's witness and initial
basis are the full-index bit-reversals of flock's `w` / `b`, under which
zorch's folds, commits, and induces reproduce flock's bytes exactly (gate:
`testing/zorch_ligerito_driver_oracle_test.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from zorch.byte_transcript import ByteHashTranscript
from zorch.hash.sha256 import HashlibSha256
from zorch.pcs.ligerito.choreography import LigeritoChoreography
from zorch.pcs.ligerito.config import LigeritoConfig

FLOCK_LIGERITO_LABEL = b"flock-ligerito-basis-v0"


def _f128_bytes(values: Array) -> bytes:
    """ghash array (any shape) -> per-element lo‖hi LE bytes, C order. The uint8
    bitcast is the one working ghash→integer direction (zorch#399)."""
    return np.asarray(lax.bitcast_convert_type(values, jnp.uint8)).tobytes()


def _f128_from_bytes(buf: bytes, n: int) -> Array:
    """n*16 squeezed bytes -> (n,) ghash (each element lo‖hi LE)."""
    u8 = jnp.asarray(np.frombuffer(buf, np.uint8).reshape(n, 16))
    return lax.bitcast_convert_type(u8, jnp.binary_field_ghash)


def _np_bytes(values: Array) -> bytes:
    return np.asarray(values).tobytes()


@dataclass(frozen=True)
class FlockTranscript:
    """zorch `Transcript` over flock's byte challenger substrate.

    `sample(1)` is a scalar squeeze and `sample(n>1)` a slice squeeze, matching
    where the driver draws flock's `sample_f128` vs `sample_f128_vec` — the two
    frame differently, so a config whose vector draws degenerate to width 1
    (`queries[j] <= 2`, which makes the alpha draw 0- or 1-wide) is rejected by
    `flock_ligerito_config` rather than silently mis-framed.
    """

    inner: ByteHashTranscript

    @property
    def has_dedicated_fusion(self) -> bool:
        return self.inner.has_dedicated_fusion

    def observe(self, values: Array) -> "FlockTranscript":
        if values.dtype == jnp.uint8:
            return FlockTranscript(self.inner.observe_bytes(_np_bytes(values)))
        if values.dtype != jnp.binary_field_ghash:
            raise TypeError(f"no flock framing for observed dtype {values.dtype}")
        inner = self.inner
        raw = _f128_bytes(values)
        for off in range(0, len(raw), 16):
            inner = inner.observe_scalar(raw[off : off + 16])
        return FlockTranscript(inner)

    def sample(self, n: int = 1) -> tuple["FlockTranscript", Array]:
        if n == 1:
            inner, buf = self.inner.sample_scalar(16)
        else:
            inner, buf = self.inner.sample_slice(n, 16)
        return FlockTranscript(inner), _f128_from_bytes(buf, n)

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple["FlockTranscript", Array]:
        return self.observe(values).sample(n)


def flock_transcript(domain: bytes) -> FlockTranscript:
    """A fresh `FlockTranscript` seeded like flock's `FsChallenger` (host
    SHA-256; flock's Fiat-Shamir is host-sequential)."""
    return FlockTranscript(ByteHashTranscript.new(domain, HashlibSha256()))


@dataclass(frozen=True)
class FlockChoreography(LigeritoChoreography):
    """flock `pcs::ligerito`'s Fiat-Shamir choreography over `FlockTranscript`.

    `fold_grinding_bits[level]` tapers per fold round (`max(bits - j, 0)`,
    ground only when > 0 — flock's conditional fold PoW);
    `query_grinding_bits[level]` is unconditional (0 bits still grinds a
    trivial nonce onto the wire, flock's convention).
    """

    fold_grinding_bits: tuple[int, ...] = ()
    query_grinding_bits: tuple[int, ...] = ()

    @property
    def eager_messages(self) -> bool:
        return True

    def bind_statement(
        self, transcript: FlockTranscript, root: Array, point: Array, value: Array
    ) -> FlockTranscript:
        del point  # flock binds the point through the outer basis, not here
        transcript = FlockTranscript(
            transcript.inner.observe_label(FLOCK_LIGERITO_LABEL)
        )
        transcript = transcript.observe(value)
        return transcript.observe(root)

    def fold_challenge(
        self, transcript: FlockTranscript, msg: Array | None, level: int, fold_idx: int
    ) -> tuple[FlockTranscript, Array]:
        del msg, level, fold_idx  # eager: the message is already absorbed
        transcript, r = transcript.sample(1)
        return transcript, r[0]

    def fold_grind_bits(self, level: int, fold_idx: int) -> int | None:
        bits = self.fold_grinding_bits[level] - fold_idx
        return bits if bits > 0 else None

    def query_grind_bits(self, level: int) -> int | None:
        return self.query_grinding_bits[level]

    def grind(
        self, transcript: FlockTranscript, bits: int
    ) -> tuple[FlockTranscript, Array]:
        inner, nonce = transcript.inner.grind_pow(bits)
        return FlockTranscript(inner), jnp.asarray(nonce, jnp.uint64)

    def check_grind(
        self, transcript: FlockTranscript, bits: int, witness: Array
    ) -> tuple[FlockTranscript, Array]:
        inner, ok = transcript.inner.verify_pow(int(witness), bits)
        return FlockTranscript(inner), jnp.asarray(ok)

    def sample_queries(
        self, transcript: FlockTranscript, block_len: int, count: int
    ) -> tuple[FlockTranscript, Array]:
        # flock `sample_distinct_queries`: one scalar F128 draw per candidate,
        # low limb mod block_len, re-draw on repeat, sorted ascending.
        inner = transcript.inner
        seen: set[int] = set()
        out: list[int] = []
        while len(out) < count:
            inner, buf = inner.sample_scalar(16)
            pos = int.from_bytes(buf[:8], "little") % block_len
            if pos not in seen:
                seen.add(pos)
                out.append(pos)
        out.sort()
        return FlockTranscript(inner), jnp.asarray(out, jnp.int32)

    def observe_residual(
        self, transcript: FlockTranscript, residual: Array
    ) -> FlockTranscript:
        # The driver's residual rides zorch's index orientation; flock's wire
        # observes it in flock's (the two are bit-reversals of each other under
        # the monomial-commit correspondence).
        return transcript.observe(lax.bit_reverse(residual, dimensions=(0,)))


def flock_ligerito_config(
    cfg: dict, log_n: int
) -> tuple[LigeritoConfig, FlockChoreography]:
    """A `dump_ligerito` config block as zorch's `(LigeritoConfig,
    FlockChoreography)`. flock's level indexing maps directly: level j of the
    zorch schedule is flock's initial commit (j=0) or recursive step j-1, so
    `fold_ks = (initial_k, *recursive_ks)` and the rate/query/grind lists carry
    over positionally; `ood_samples[0]` is required 0 by flock's verifier and
    zorch's schedule starts at the first recursive commit (`ood[1:]`)."""
    fold_ks = (cfg["initial_k"], *cfg["recursive_ks"])
    num_levels = len(fold_ks)
    queries = tuple(cfg["queries"])
    if any(q <= 2 for q in queries):
        raise ValueError(
            f"queries {queries}: a level with <= 2 queries makes its alpha draw "
            "0- or 1-wide, where flock's slice framing and FlockTranscript's "
            "scalar/slice split diverge"
        )
    ood = tuple(cfg["ood_samples"])
    if ood and ood[0] != 0:
        raise ValueError(f"flock requires ood_samples[0] == 0, got {ood}")
    config = LigeritoConfig(
        num_vars=log_n,
        fold_ks=fold_ks,
        log_inv_rates=tuple(cfg["log_inv_rates"])[:num_levels],
        queries=queries[:num_levels],
        ood_samples=ood[1:num_levels],
        alpha_lsb_first=True,
        compressed_sumcheck_messages=True,
        monomial_commit=True,
    )
    choreography = FlockChoreography(
        fold_grinding_bits=tuple(cfg["fold_grinding_bits"])[:num_levels],
        query_grinding_bits=tuple(cfg["grinding_bits"])[:num_levels],
    )
    return config, choreography
