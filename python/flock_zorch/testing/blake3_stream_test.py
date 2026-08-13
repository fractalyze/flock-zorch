# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`hash.blake3_stream` against hash-frx's one-shot BLAKE3.

The property under test is the only one a streaming state has to have:
**absorbing a message in any sequence of pieces equals hashing it whole.** The
oracle is `HostBlake3`, the same implementation `Blake3ByteTranscript` already
hashes with, so a divergence here is this module's and not the fork's.

Why this exists: the BLAKE3 arm's first transcript kept its state on the host
behind `io_callback`, which made it un-threadable through a jitted loop — the
sumcheck round loop then de-compiled into a host loop, worth ~10x of the
snark.fast window at m32. A device transcript fixed that, and it needs a
resumable hash state whose SHAPE does not depend on how many bytes have been
absorbed. Hence a streaming state rather than a growing buffer: inside a loop
the absorbed length is a runtime value, and a one-shot digest needs a static
message shape.

The split patterns matter more than the sizes. BLAKE3 is a tree hash over
1024-byte chunks with a 64-byte block inside each, so the interesting cases are
the ones that land a piece boundary exactly on a block edge, exactly on a chunk
edge, and one byte either side of both.
"""

import unittest

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

from hash_frx.blake3.byte_hashes import HostBlake3  # noqa: E402

from flock_zorch.hash import blake3_stream  # noqa: E402

BLOCK = 64
CHUNK = 1024


def _host(msg: bytes, out_len: int = 32) -> bytes:
    row = np.frombuffer(msg, np.uint8)[None, :]
    return bytes(np.asarray(HostBlake3(out_len).digest(row))[0])


def _device(pieces, out_len: int = 32) -> bytes:
    """Absorb `pieces` in order into a fresh state, then read `out_len` bytes.

    Jitted as one program: absorbing eagerly compiles every `lax` primitive
    inside `absorb` separately, which for a long split is minutes of tracing for
    microseconds of hashing.
    """

    @frx.jit
    def run(*parts):
        st = blake3_stream.init()
        for part in parts:
            st = blake3_stream.absorb(st, part)
        return blake3_stream.finalize(st, out_len)

    arrays = [frx.device_put(np.frombuffer(p, np.uint8)) for p in pieces]
    return bytes(np.asarray(run(*arrays)))


def _splits(msg: bytes):
    """Piece sequences worth trying for a message of this length."""
    n = len(msg)
    out = [[msg]]  # one shot
    for cut in {1, BLOCK - 1, BLOCK, BLOCK + 1, CHUNK - 1, CHUNK, CHUNK + 1, n // 2}:
        if 0 < cut < n:
            out.append([msg[:cut], msg[cut:]])
    # Many small pieces exercises the partial-block path repeatedly, but each
    # distinct piece length is its own trace, so this is the expensive arm —
    # keep it to messages short enough that it covers the partial-block cases
    # without paying for them at every size.
    if 3 < n <= 128:
        out.append([msg[i : i + 7] for i in range(0, n, 7)])
    return out


class Blake3StreamTest(unittest.TestCase):
    # One length per class the tree hash distinguishes: empty, sub-block, the
    # 64-byte block edge either side, the 1024-byte chunk edge either side, and
    # the 24,163 bytes a full m32 prove absorbs (multi-chunk, so it is the only
    # one that exercises the subtree-stack merge). Sizes between those classes
    # cost a trace each and test nothing new.
    LENGTHS = (0, 1, 63, 64, 65, 1023, 1024, 1025, 24163)

    def test_matches_host_one_shot(self):
        rng = np.random.default_rng(0)
        for n in self.LENGTHS:
            msg = rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
            want = _host(msg)
            for pieces in _splits(msg):
                with self.subTest(n=n, pieces=[len(p) for p in pieces]):
                    self.assertEqual(_device(pieces).hex(), want.hex())

    def test_xof_read_lengths(self):
        """The transcript squeezes non-32-byte lengths (a 16-byte F128 draw is
        the common one), so the extendable read is part of the contract."""
        rng = np.random.default_rng(1)
        msg = rng.integers(0, 256, size=200, dtype=np.uint8).tobytes()
        for out_len in (1, 16, 31, 32, 33, 64, 96):
            with self.subTest(out_len=out_len):
                self.assertEqual(
                    _device([msg], out_len).hex(), _host(msg, out_len).hex()
                )

    def test_state_shape_is_absorb_invariant(self):
        """The point of the whole exercise: the state's pytree structure must not
        depend on how much has been absorbed, or it cannot be a loop carry."""
        st0 = blake3_stream.init()
        st1 = blake3_stream.absorb(st0, frx.device_put(np.zeros(3, np.uint8)))
        st2 = blake3_stream.absorb(st1, frx.device_put(np.zeros(5000, np.uint8)))
        base = frx.tree_util.tree_structure(st0)
        self.assertEqual(frx.tree_util.tree_structure(st1), base)
        self.assertEqual(frx.tree_util.tree_structure(st2), base)
        for a, b in zip(frx.tree_util.tree_leaves(st0), frx.tree_util.tree_leaves(st2)):
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)

    def test_threads_a_jitted_loop(self):
        """What the transcript actually needs: survive `lax.fori_loop` as a carry.
        If this compiles and matches, the round loop can stay in the program."""
        piece = np.arange(100, dtype=np.uint8)

        @frx.jit
        def run(p):
            st = blake3_stream.init()
            st = frx.lax.fori_loop(0, 10, lambda _, s: blake3_stream.absorb(s, p), st)
            return blake3_stream.finalize(st, 32)

        got = bytes(np.asarray(run(frx.device_put(piece))))
        self.assertEqual(got.hex(), _host(piece.tobytes() * 10).hex())


if __name__ == "__main__":
    unittest.main()
