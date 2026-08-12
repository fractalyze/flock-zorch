# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`Blake3FieldTranscript` against the host `Blake3Challenger`.

The host arm is already pinned to flock-challenge's own `FsChallenger` by
`blake3_challenger_test`, so matching it here transitively pins the device arm
to the fork — which is what lets `BENCHMARK_PROFILE` swap onto it without the
wire moving.

Fiat-Shamir makes this gate sharp for free: every draw binds the whole prefix,
so a single wrong byte anywhere shows up as a different challenge at the next
sample and never cancels out. The scripted sequence therefore interleaves all
five observe kinds with both sample kinds, and the last draw is the one that
would catch drift introduced by any earlier op.

Scalar-vs-slice framing is the trap worth naming: `sample_scalar()` and
`sample(1)` differ on the wire (the KIND tag), as do `observe_scalar` and
`observe` of one element. Both pairs are exercised.
"""

import unittest

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402

from flock_zorch import ghash  # noqa: E402
from flock_zorch.blake3_challenger import Blake3Challenger  # noqa: E402
from flock_zorch.hash.blake3_field_transcript import (  # noqa: E402
    Blake3FieldTranscript,
)

DOMAIN = b"flock-bench-v0"


def _g(vals):
    """Host F128 elements from python ints."""
    lanes = np.array([[v & ((1 << 64) - 1), v >> 64] for v in vals], dtype=np.uint64)
    return ghash._lanes_to_ghash(lanes)


class Blake3FieldTranscriptTest(unittest.TestCase):
    def _script(self, host, dev, sink):
        """Run one op sequence on both arms, appending every draw to `sink`.

        `host` is mutable (`&mut self`) and `dev` is functional, so the device
        arm is rebound at every step; returning it keeps that explicit.
        """
        host.observe_label(b"flock-r1cs-v0")
        dev = dev.observe_label(b"flock-r1cs-v0")

        root = np.arange(32, dtype=np.uint8)
        host.observe_bytes(root)
        dev = dev.observe_bytes(frx.device_put(root))

        one = _g([0x0123_4567_89AB_CDEF])
        host.observe_f128(one[0])  # 0-d: scalar framing
        dev = dev.observe_scalar(frx.device_put(one)[0])

        host_s = host.sample_f128()  # scalar draw
        dev, dev_s = dev.sample_scalar()
        sink.append(("scalar", host_s, dev_s))

        many = _g([1, 2, 3, 1 << 100])
        host.observe_f128(many)  # 1-D: slice framing
        dev = dev.observe(frx.device_put(many))

        host_v = host.sample_f128(3)  # slice draw
        dev, dev_v = dev.sample(3)
        sink.append(("slice3", host_v, dev_v))

        # slice-of-one is NOT the scalar draw — different KIND tag on the wire
        host_v1 = host.sample_f128(1)
        dev, dev_v1 = dev.sample(1)
        sink.append(("slice1", host_v1, dev_v1))

        host.observe_label(b"tail")
        dev = dev.observe_label(b"tail")
        host_last = host.sample_f128()
        dev, dev_last = dev.sample_scalar()
        sink.append(("after-all", host_last, dev_last))
        return dev

    def test_matches_host_challenger(self):
        host = Blake3Challenger(DOMAIN)
        dev = Blake3FieldTranscript.new(DOMAIN, fnp.binary_field_ghash)
        draws = []
        self._script(host, dev, draws)
        for name, want, got in draws:
            with self.subTest(draw=name):
                np.testing.assert_array_equal(
                    ghash._ghash_to_lanes(np.asarray(want)),
                    ghash._ghash_to_lanes(np.asarray(got)),
                )

    def test_scalar_and_slice_of_one_differ(self):
        """A guard on the gate itself: if these agreed, the test above would pass
        with the KIND tag ignored."""
        a = Blake3FieldTranscript.new(DOMAIN, fnp.binary_field_ghash)
        _, scalar = a.sample_scalar()
        _, slice1 = Blake3FieldTranscript.new(DOMAIN, fnp.binary_field_ghash).sample(1)
        self.assertFalse(
            np.array_equal(
                ghash._ghash_to_lanes(np.asarray(scalar)),
                ghash._ghash_to_lanes(np.asarray(slice1))[0],
            )
        )

    def test_grind_matches_host(self):
        """PoW: the fork pads its pre-image to a whole 64-byte block, which the
        generic byte transcript does not — so this is a distinct wire, and the
        nonce must agree exactly (it is the lowest hit, not any hit)."""
        for bits in (0, 4, 9):
            with self.subTest(bits=bits):
                host = Blake3Challenger(DOMAIN)
                dev = Blake3FieldTranscript.new(DOMAIN, fnp.binary_field_ghash)
                host.observe_label(b"pow")
                dev = dev.observe_label(b"pow")
                want = host.grind_pow(bits)
                dev, got = dev.grind(bits)
                self.assertEqual(int(got), want)
                # the nonce is absorbed either way, so the next draw pins it
                nxt_host = host.sample_f128()
                dev, nxt_dev = dev.sample_scalar()
                np.testing.assert_array_equal(
                    ghash._ghash_to_lanes(np.asarray(nxt_host)),
                    ghash._ghash_to_lanes(np.asarray(nxt_dev)),
                )

    def test_threads_a_jitted_loop(self):
        """The reason the module exists: usable as a `fori_loop` carry, so a
        round loop stays inside the compiled program."""

        @frx.jit
        def run():
            t = Blake3FieldTranscript.new(DOMAIN, fnp.binary_field_ghash)

            def body(_, carry):
                t_, acc = carry
                t_, c = t_.sample_scalar()
                return t_, acc + c

            _, acc = frx.lax.fori_loop(
                0, 8, body, (t, fnp.zeros((), fnp.binary_field_ghash))
            )
            return acc

        host = Blake3Challenger(DOMAIN)
        want = _g([0])[0]
        for _ in range(8):
            want = want + host.sample_f128()
        np.testing.assert_array_equal(
            ghash._ghash_to_lanes(np.asarray(run())),
            ghash._ghash_to_lanes(np.asarray(want)),
        )


if __name__ == "__main__":
    unittest.main()
