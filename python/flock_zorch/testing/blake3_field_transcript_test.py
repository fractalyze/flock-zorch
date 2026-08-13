# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""zorch's `Blake3FieldTranscript`, as THIS repo wires it, against the host
`Blake3Challenger`.

The transcript itself is zorch's and is gated there against its own byte oracle.
What that cannot check is the wiring: the fork pads its proof-of-work pre-image
to a whole block where zorch's default stops at 40, so the device arm is only
the fork's arm if `_initial_device_transcript` says so. Every device transcript
below therefore comes from that function rather than from a bare `new` — a test
that constructed its own would pass while the prover spoke a different wire.

The host arm is already pinned to flock-challenge's own `FsChallenger` by
`blake3_challenger_test`, so matching it here transitively pins the device arm
to the fork — which is what lets `BLAKE3_PROFILE` swap onto it without the
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
from flock_zorch.blake3_challenger import (  # noqa: E402
    Blake3Challenger,
    _initial_device_transcript,
)

DOMAIN = b"flock-bench-v0"


def _dev():
    """The device transcript the prover actually runs — the fork's PoW width
    included. Never `Blake3FieldTranscript.new` directly here: zorch's default
    is the unpadded wire, so a bare construction would gate the wrong arm."""
    return _initial_device_transcript(DOMAIN)


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
        dev = _dev()
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
        a = _dev()
        _, scalar = a.sample_scalar()
        _, slice1 = _dev().sample(1)
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
                dev = _dev()
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


class RoundLoopTest(unittest.TestCase):
    """The leading indicator: the sumcheck round loop must compile INTO the
    prove program under the BLAKE3 arm.

    Whether the loop is in the program is a tracing property, so this needs no
    golden and no GPU — `lower` traces, it does not execute, so synthetic
    witness shapes carry the same program structure and the check runs in CPU
    CI. Check it before any wall-clock claim: a transcript that cannot be a
    loop carry drops the count to zero and costs ~10x at m32, which is what the
    retired host-callback arm did.

    The SHA-256 arm is the control, exactly as in `blake3_challenger_test` — a
    red BLAKE3 count with a green control is a BLAKE3-arm regression, both red
    is the zerocheck itself losing its loops.
    """

    M = 22

    def _whiles(self, challenger_cls):
        from flock_zorch import zerocheck

        rows = 1 << (self.M - 7)
        packed = fnp.zeros((rows, 2), fnp.uint64)
        ch = challenger_cls(DOMAIN)

        def run(a, b, c):
            proof, _ = zerocheck.prove_packed(a, b, c, self.M, ch=ch)
            return proof.round1_ab, proof.round1_c

        return frx.jit(run).lower(packed, packed, packed).as_text().count("while(")

    def test_blake3_arm_keeps_the_round_loop(self):
        from flock_zorch.blake3_challenger import Blake3DeviceChallenger

        self.assertGreater(
            self._whiles(Blake3DeviceChallenger),
            0,
            "the BLAKE3 arm's zerocheck lowered with no `while` loop — its "
            "transcript stopped being usable as a loop carry, so the round "
            "loop de-compiled into a host loop",
        )

    def test_sha256_control_keeps_the_round_loop(self):
        from flock_zorch.sha256_challenger import Sha256Challenger

        self.assertGreater(self._whiles(Sha256Challenger), 0)


class Blake3ProfileTest(unittest.TestCase):
    """`BLAKE3_PROFILE` is composed of the device arm and the BLAKE3 tree.

    `prove_phase_bench_test` pins that `--hash blake3` SELECTS this profile;
    this pins what the profile IS. Without it, swapping the challenger back to
    a host-backed one would leave every gate green while the prove lost its
    round loop.
    """

    def test_profile_is_the_device_arm_and_the_blake3_tree(self):
        from flock_zorch import prover
        from flock_zorch.blake3_challenger import Blake3DeviceChallenger
        from flock_zorch.hash import merkle

        self.assertIs(prover.BLAKE3_PROFILE.challenger_cls, Blake3DeviceChallenger)
        self.assertIs(prover.BLAKE3_PROFILE.tree, merkle.GHASH_BLAKE3_TREE)


if __name__ == "__main__":
    unittest.main()
