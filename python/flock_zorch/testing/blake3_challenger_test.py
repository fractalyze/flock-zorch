# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Fork-fixture byte gate for the BLAKE3 Fiat-Shamir challenger.

The fixture below was printed by the flock-challenge fork's own
`FsChallenger` (Layr-Labs/flock-challenge @ d86604387a795f74aa526176f0911763
df256405 — the crate the benchmark harness verifies with), driven through
exactly the absorb/sample sequence `_drive` performs, once per hash arm:
`with_hash(b"flock-bench-v0", Blake3)` and `with_hash(_, Sha256)`. To
regenerate, run that sequence through the fork's `flock-core` crate
(a ~60-line `main.rs` path-depping `crates/flock-core`; print each draw as
the F128 wire bytes `lo_le8 ‖ hi_le8`, each nonce as its integer) and paste
the hex here.

Three pins, in dependency order:

- **The BLAKE3 arm matches the fork** — every draw byte-for-byte, both grind
  nonces, and the post-grind draw. This is what lets a benchmark-profile
  proof pass their verifier's first transcript draw.
- **The verifier mirror stays in lockstep**: `verify_pow` accepts the ground
  nonces, rejects a wrong one, and REGARDLESS lands on the prover's
  post-grind state (the fork absorbs the nonce even on a failed check).
- **The SHA-256 control**: the existing device challenger, driven through
  the same sequence, matches the fork's SHA-256 arm. This isolates failures —
  a red BLAKE3 gate with a green control is a BLAKE3-specific bug, not a
  sequence-mapping one — and pins the SHA arm against the fork directly
  (the proof goldens pin it only transitively).

The BLAKE3 grind fixture (nonce 4541) deliberately crosses the challenger's
4096-nonce grind window, so the multi-window scan path is exercised.

The host row runs on numpy; the device rows (BLAKE3 and the SHA-256 control)
run eager device transcript hops, which are CPU-safe.
"""
from __future__ import annotations

import frx
import numpy as np
from absl.testing import absltest

from flock_zorch.blake3_challenger import (
    Blake3Challenger,
    Blake3DeviceChallenger,
    _initial_device_transcript,
)
from flock_zorch.ghash import _lanes_to_ghash, to_lanes
from flock_zorch.sha256_challenger import Sha256Challenger

_DOMAIN = b"flock-bench-v0"
_LABEL = b"flock-zerocheck-v0"
_OBS_SCALAR = (0x0123456789ABCDEF, 0xFEDCBA9876543210)
_OBS_SLICE = ((1, 2), (3, 4), (5, 6))
_ROOT = b"\xaa" * 32

# Fork draws for `_drive`'s sequence (see the module docstring for provenance).
_BLAKE3 = {
    "s1": "61d9d8dcc367662cd947b31b4d8c5a08",
    "v1": "9eea0fc66bdc7051487cb670cb6ac054",
    "v2": "4f5fac7939546e745b8d15091db8db1b7a838939135fffe8f192488d1366fb9d",
    "v5": "454d9206c85fa074b9a02bc76b79a8b879bab24f3d55ebde45998c3845f5c3c8"
    "55f939097d5b5359e6b98eb97bc87e5a51bbfc87f7595df2ad8f431d803f7734"
    "3b1e4c5ca3581c71460c63afa90d432a",
    "s2": "681e019ba4711c5edf9652261f66765e",
    "n0": 0,
    "n12": 4541,
    "s3": "5cac776210ab68d8b56c7d3bcd348703",
}
_SHA256 = {
    "s1": "b66153a705789f5d8c592844acaeaa88",
    "v1": "1b51872b8ea0f7b280115c916e3b896b",
    "v2": "4a1882999d4361eda3768e8b02fccfcd0a7fdbd128065ca2a8ba5af16ffcb6ce",
    "v5": "6e4e49ae265fbb8cb779e3b4767d8c304d9e6b6c28b9d9f55c7cff8b7d23364f"
    "ad279388420754a32897285cd7580209c2fdac68467be6ee58bc98bec77d3963"
    "96d1b34b9ec537e6a38b1eec7daa2b51",
    "s2": "c8925d24e9f30d48010246cf5361d488",
    "n0": 0,
    "n12": 6389,
    "s3": "51d283b92d6b07ac2591ee78e04d7113",
}


def _g(lanes):
    """Host ghash from (lo, hi) lane tuples — scalar for one pair, slice for a
    tuple of pairs."""
    return _lanes_to_ghash(np.asarray(lanes, dtype=np.uint64))


def _wire(g) -> str:
    """A draw's wire hex: lo_le8 ‖ hi_le8 per element, elements in order."""
    return to_lanes(g).tobytes().hex()


def _drive(challenger) -> dict:
    """The fixture's scripted sequence — keep in lockstep with the dump."""
    out = {}
    challenger.observe_label(_LABEL)
    challenger.observe_f128(_g(_OBS_SCALAR))
    out["s1"] = _wire(challenger.sample_f128())
    challenger.observe_f128(_g(_OBS_SLICE))
    out["v1"] = _wire(challenger.sample_f128(1))  # slice(1) framing, not scalar
    out["v2"] = _wire(challenger.sample_f128(2))  # exactly one 32 B digest of stream
    out["v5"] = _wire(challenger.sample_f128(5))  # 80 B, beyond one digest
    challenger.observe_bytes(_ROOT)
    out["s2"] = _wire(challenger.sample_f128())
    out["n0"] = challenger.grind_pow(0)
    out["n12"] = challenger.grind_pow(12)
    out["s3"] = _wire(challenger.sample_f128())
    return out


def _replay_observes(challenger) -> None:
    """`_drive` up to (not including) the grinds — the verifier's replay."""
    challenger.observe_label(_LABEL)
    challenger.observe_f128(_g(_OBS_SCALAR))
    challenger.sample_f128()
    challenger.observe_f128(_g(_OBS_SLICE))
    challenger.sample_f128(1)
    challenger.sample_f128(2)
    challenger.sample_f128(5)
    challenger.observe_bytes(_ROOT)
    challenger.sample_f128()


class Blake3ChallengerForkGateTest(absltest.TestCase):
    def test_every_draw_matches_the_fork(self):
        got = _drive(Blake3Challenger(_DOMAIN))
        self.assertEqual(got, _BLAKE3)

    def test_verify_pow_mirror_stays_in_lockstep(self):
        c = Blake3Challenger(_DOMAIN)
        _replay_observes(c)
        self.assertTrue(c.verify_pow(_BLAKE3["n0"], bits=0))
        self.assertTrue(c.verify_pow(_BLAKE3["n12"], bits=12))
        self.assertEqual(_wire(c.sample_f128()), _BLAKE3["s3"])

    def test_verify_pow_rejects_a_wrong_nonce(self):
        c = Blake3Challenger(_DOMAIN)
        _replay_observes(c)
        self.assertTrue(c.verify_pow(0, bits=0))
        # Nonce 0 cannot satisfy the 12-bit grind here: the prover's grind
        # returns the SMALLEST satisfying nonce and it ground to 4541.
        self.assertFalse(c.verify_pow(0, bits=12))

    def test_zero_bit_site_accepts_only_the_canonical_nonce(self):
        # Malleability guard, same as the fork: bits == 0 still requires the
        # canonical nonce 0.
        c = Blake3Challenger(_DOMAIN)
        _replay_observes(c)
        self.assertFalse(c.verify_pow(1, bits=0))


@frx.jit
def _fixture_zone(t, x_scalar, x_slice, root):
    """The whole fixture sequence as ONE jitted program — every op a device
    computation on the threaded transcript state."""
    t = t.observe_label(_LABEL)
    t = t.observe_scalar(x_scalar)
    t, s1 = t.sample_scalar()
    t = t.observe(x_slice)
    t, v1 = t.sample(1)
    t, v2 = t.sample(2)
    t, v5 = t.sample(5)
    t = t.observe_bytes(root)
    t, s2 = t.sample_scalar()
    t, n0 = t.grind(0)
    t, n12 = t.grind(12)
    t, s3 = t.sample_scalar()
    return t, s1, v1, v2, v5, s2, n0, n12, s3


class DeviceTranscriptForkGateTest(absltest.TestCase):
    """The prove-path arm: the device transcript through jitted zones. The
    fixture pin here is what licenses threading it through the four
    transcript-carrying zones, and it pins the device row to the fork
    DIRECTLY — `blake3_field_transcript_test` only reaches the fork through
    the host row."""

    def test_one_jitted_zone_matches_the_fork(self):
        t = _initial_device_transcript(_DOMAIN)
        root = np.frombuffer(_ROOT, np.uint8)
        _, s1, v1, v2, v5, s2, n0, n12, s3 = _fixture_zone(
            t, _g(_OBS_SCALAR), _g(_OBS_SLICE), root
        )
        got = {
            "s1": _wire(s1),
            "v1": _wire(v1),
            "v2": _wire(v2),
            "v5": _wire(v5),
            "s2": _wire(s2),
            "n0": int(n0),
            "n12": int(n12),
            "s3": _wire(s3),
        }
        self.assertEqual(got, _BLAKE3)

    def test_eager_wrapper_matches_the_fork(self):
        # The same ops OUTSIDE any jit zone — the eager touchpoints' path.
        got = _drive(Blake3DeviceChallenger(_DOMAIN))
        self.assertEqual(got, _BLAKE3)

    def test_scan_body_matches_the_eager_challenger(self):
        # fs.sample_chain's shape: sample_scalar as a lax.scan body.
        ref = Blake3Challenger(_DOMAIN)
        ref.observe_label(_LABEL)
        want = [_wire(ref.sample_f128()) for _ in range(3)]

        @frx.jit
        def chain(t):
            t = t.observe_label(_LABEL)
            return frx.lax.scan(lambda t, _: t.sample_scalar(), t, None, length=3)

        _, draws = chain(_initial_device_transcript(_DOMAIN))
        got = [_wire(np.asarray(draws)[i]) for i in range(3)]
        self.assertEqual(got, want)

    def test_check_witness_mirror(self):
        c = Blake3DeviceChallenger(_DOMAIN)
        c.observe_label(_LABEL)
        nonce = c.grind_pow(12)
        v = Blake3DeviceChallenger(_DOMAIN)
        v.observe_label(_LABEL)
        v2, ok = v.check_witness(np.uint64(nonce), pow_bits=12)
        self.assertTrue(bool(np.asarray(ok)))
        # Post-check lockstep: both sides draw the same next challenge.
        self.assertEqual(_wire(c.sample_f128()), _wire(v2.sample_f128()))


class Sha256ControlForkGateTest(absltest.TestCase):
    """The existing SHA-256 challenger through the same sequence. A red BLAKE3
    gate with this green is a BLAKE3-arm bug; both red is a sequence-mapping
    bug in `_drive`."""

    def test_every_draw_matches_the_fork(self):
        got = _drive(Sha256Challenger(_DOMAIN))
        self.assertEqual(got, _SHA256)


if __name__ == "__main__":
    absltest.main()
