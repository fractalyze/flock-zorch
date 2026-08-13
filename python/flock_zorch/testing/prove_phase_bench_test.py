# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`prove_phase_bench`'s `--hash` selects a whole arm, and says which it picked.

Both matter for the same reason: a wall-clock number from this harness is
meaningless without knowing which Fiat-Shamir and Merkle hashes produced it, and
for a while the harness could only run one arm and never named it. A number
attributed to the wrong arm survives into an issue and costs a day.
"""

import unittest

import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch import prover  # noqa: E402
from flock_zorch.testing import prove_phase_bench as bench  # noqa: E402


class HashArmTest(unittest.TestCase):
    def test_each_arm_resolves_to_a_distinct_profile(self):
        seen = {}
        for name, attr in bench.HASH_ARMS.items():
            profile = getattr(prover, attr)
            seen[name] = (profile.challenger_cls, type(profile.tree))
        self.assertEqual(len(set(seen.values())), len(seen), f"arms collide: {seen}")

    def test_the_names_mean_what_they_say(self):
        """`sha256` must stay flock's arm — every golden byte-gates it, so a
        silent switch would compare new numbers against a different protocol."""
        self.assertIs(getattr(prover, bench.HASH_ARMS["sha256"]), prover.SHA256_PROFILE)
        self.assertIs(getattr(prover, bench.HASH_ARMS["blake3"]), prover.BLAKE3_PROFILE)

    def test_profile_reaches_the_prove(self):
        """`_profile` is what `make_prove` threads into commit and open; if the
        flag stopped reaching it the harness would silently measure one arm
        under both names."""

        class Args:
            hash = "blake3"

        self.assertIs(bench._profile(Args()), prover.BLAKE3_PROFILE)
        Args.hash = "sha256"
        self.assertIs(bench._profile(Args()), prover.SHA256_PROFILE)


if __name__ == "__main__":
    unittest.main()
