# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The fingerprint parsers and the drift verdict, against recorded accidents.

Every case here is a state this repo has actually been measured in, and in
each one the run produced a plausible number rather than an error — which is
why the check has to be mechanical. `docs/measurement.md` narrates them; this
file is the executable form, so a regression in the parser shows up as a red
test instead of as a wrong benchmark months later.

Pure text and dicts — no GPU, no subprocess — so the accidents are reproduced
without a box that reproduces the accident.
"""

from __future__ import annotations

from typing import Any

from absl.testing import absltest

from flock_zorch.testing._fingerprint import (
    BLOCKING,
    INFORMATIONAL,
    blocking,
    drift,
    parse_module_pins,
    parse_override_modules,
    parse_requirement_pins,
    publish_refusal,
    release,
)

# The shape `.bazelrc.user` ships with: the override is documented and inert.
_BAZELRC_COMMENTED = """\
# DEV MODE — to build against local zorch edits, uncomment the line below.
#
# common --override_module=zorch=/home/ryan/Workspace/envs/flock-zorch/zorch
"""
# The state the #200 erratum was measured in.
_BAZELRC_ACTIVE = """\
# DEV MODE
common --override_module=zorch=/home/ryan/Workspace/envs/flock-zorch/zorch
"""

_MODULE_BAZEL = """\
bazel_dep(name = "zorch", version = "0.0.0")
git_override(
    module_name = "zorch",
    commit = "b3003e7cb234abede44b8b3ab46c5269e8165f37",
    remote = "https://github.com/fractalyze/zorch.git",
)
"""


class ParseOverrideModulesTest(absltest.TestCase):
    """A commented override is inert; an active one substitutes the source.

    Reading the two the same way is what makes a benchmark quote a number
    produced by a checkout the repo never declared.
    """

    def test_commented_override_is_not_active(self) -> None:
        self.assertEqual(parse_override_modules(_BAZELRC_COMMENTED), {})

    def test_active_override_is_found(self) -> None:
        self.assertEqual(
            parse_override_modules(_BAZELRC_ACTIVE),
            {"zorch": "/home/ryan/Workspace/envs/flock-zorch/zorch"},
        )

    def test_indented_override_is_active(self) -> None:
        # Leading whitespace does not comment a line out.
        self.assertEqual(
            parse_override_modules("  common --override_module=zorch=/tmp/z"),
            {"zorch": "/tmp/z"},
        )

    def test_empty_bazelrc_has_no_overrides(self) -> None:
        # The file is optional; a missing one reads as "" and must not throw.
        self.assertEqual(parse_override_modules(""), {})


class ParsePinsTest(absltest.TestCase):
    def test_git_override_commit(self) -> None:
        self.assertEqual(
            parse_module_pins(_MODULE_BAZEL),
            {"zorch": "b3003e7cb234abede44b8b3ab46c5269e8165f37"},
        )

    def test_requirement_pins_normalise_the_distribution_name(self) -> None:
        # `hash-frx` and `hash_frx` name one distribution; the record keys on
        # one spelling so a comparison across them cannot read as drift.
        pins = parse_requirement_pins("hash_frx==0.2.0\nfrx==0.10.2\n# comment\n")
        self.assertEqual(pins, {"hash-frx": "0.2.0", "frx": "0.10.2"})

    def test_release_reads_a_real_banner(self) -> None:
        self.assertEqual(
            release("Cuda compilation tools, release 13.3, V13.3.33"), "13.3"
        )
        self.assertIsNone(release(None))
        self.assertIsNone(release("not a banner"))


class DriftTest(absltest.TestCase):
    """Which changes make two numbers incomparable."""

    def test_same_fingerprint_has_no_drift(self) -> None:
        fp = {"toolchain": {"ptxas": "13.3"}, "device": {"name": "RTX 5090"}}
        self.assertEqual(drift(fp, fp), [])

    def test_toolchain_change_blocks(self) -> None:
        # The 5.5-16x variable. A delta across it measures the toolchain.
        before = {"toolchain": {"ptxas": "13.3", "nvlink": "13.3"}}
        after = {"toolchain": {"ptxas": "12.9", "nvlink": "13.3"}}
        found = blocking(drift(before, after))
        self.assertEqual([d.path for d in found], ["toolchain.ptxas"])

    def test_override_appearing_blocks(self) -> None:
        # #200: a stale override hid a 35% m32 difference, and every wall
        # measured under it had to be thrown away.
        before: dict = {"overrides": {}}
        after = {"overrides": {"zorch": {"path": "/tmp/z", "matches_pin": False}}}
        self.assertTrue(blocking(drift(before, after)))

    def test_allocator_change_blocks(self) -> None:
        # cuda_async inflates the m32 prove ~14%.
        before = {"runtime": {"XLA_PYTHON_CLIENT_ALLOCATOR": None}}
        after = {"runtime": {"XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async"}}
        found = blocking(drift(before, after))
        self.assertEqual(
            [d.path for d in found], ["runtime.XLA_PYTHON_CLIENT_ALLOCATOR"]
        )

    def test_hand_built_wheel_in_the_venv_blocks(self) -> None:
        # The venv disagreeing with the lock file is the state where the
        # number describes something the repo never declared.
        before = {"pins": {"installed": {"frx": "0.10.2.dev20260821142028"}}}
        after = {"pins": {"installed": {"frx": "0.10.2.dev20260821142028+nonroot"}}}
        self.assertTrue(blocking(drift(before, after)))

    def test_window_change_blocks(self) -> None:
        # "3.67M includes serialization" and "this lineage is the SHA-256
        # window" were footnotes carried in someone's head; barriered runs
        # ~14% slower than throughput and is for attribution only.
        before = {"window": {"mode": "throughput", "runs": 10}}
        after = {"window": {"mode": "barriered", "runs": 10}}
        self.assertTrue(blocking(drift(before, after)))

    def test_source_change_is_never_drift(self) -> None:
        # The commit under test is the independent variable: if it counted as
        # drift, every comparison worth making would be refused.
        before = {"source": {"sha": "a" * 40, "dirty": False}}
        after = {"source": {"sha": "b" * 40, "dirty": True}}
        self.assertEqual(blocking(drift(before, after)), [])

    def test_driver_change_blocks(self) -> None:
        # Two hosts can carry the same card model and differ only here — the
        # bench box runs 580.126.09, the dev box 595.84 — so without this a
        # number from one would read as comparable against the other's.
        before = {"device": {"name": "RTX 5090", "driver": "595.84"}}
        after = {"device": {"name": "RTX 5090", "driver": "580.126.09"}}
        found = blocking(drift(before, after))
        self.assertEqual([d.path for d in found], ["device.driver"])

    def test_card_count_is_informational(self) -> None:
        # The bench pins CUDA_VISIBLE_DEVICES, so what matters is which card
        # was used (recorded in device.name/pinned_to), not how many the host
        # happens to have.
        before = {"device": {"count": 1}}
        after = {"device": {"count": 2}}
        self.assertEqual([d.severity for d in drift(before, after)], [INFORMATIONAL])

    def test_unknown_field_fails_open(self) -> None:
        # A field added to the record later must not silently start refusing
        # every comparison before anyone classifies it.
        before = {"something_new": 1}
        after = {"something_new": 2}
        self.assertEqual(drift(before, after)[0].severity, INFORMATIONAL)

    def test_added_field_is_drift_against_a_record_without_it(self) -> None:
        # An older record simply lacks the key; that is a real difference and
        # must not read as equal.
        found = drift({"toolchain": {}}, {"toolchain": {"ptxas": "13.3"}})
        self.assertEqual(
            [(d.path, d.before, d.after, d.severity) for d in found],
            [("toolchain.ptxas", None, "13.3", BLOCKING)],
        )

    def test_describe_names_both_sides(self) -> None:
        found = drift({"toolchain": {"ptxas": "13.3"}}, {"toolchain": {"ptxas": None}})
        self.assertIn("13.3", found[0].describe())


class PublishRefusalTest(absltest.TestCase):
    """What an unattended run must refuse to add to a trajectory.

    Stricter than the interactive guard on purpose: nobody is watching, and a
    capped toolchain publishes a plausible number rather than an error.
    """

    def _fp(self, **toolchain: Any) -> dict[str, Any]:
        base = {
            "ptxas": "13.3",
            "nvlink": "13.3",
            "clmad_ptxas_refusal": None,
            "clmad_nvlink_refusal": None,
        }
        return {"toolchain": {**base, **toolchain}}

    def test_good_toolchain_publishes(self) -> None:
        self.assertIsNone(publish_refusal(self._fp()))

    def test_capped_ptxas_is_refused(self) -> None:
        reason = publish_refusal(
            self._fp(ptxas="12.9", clmad_ptxas_refusal="ptxas 12.9 cannot assemble")
        )
        self.assertIn("12.9", reason)

    def test_capped_nvlink_is_refused(self) -> None:
        self.assertIsNotNone(
            publish_refusal(self._fp(clmad_nvlink_refusal="nvlink 12.9 is older"))
        )

    def test_unidentified_toolchain_is_refused(self) -> None:
        # The interactive guard waves this through (an unreadable probe never
        # blocks a human). Unattended, "no evidence" is not "fine".
        reason = publish_refusal(self._fp(ptxas=None))
        self.assertIn("ptxas", reason)

    def test_old_card_still_publishes(self) -> None:
        # No clmad at any toolchain there, so the fallback IS the measurement
        # and both versions still read — refusing would block the box forever.
        self.assertIsNone(publish_refusal(self._fp(ptxas="12.9", nvlink="12.9")))


if __name__ == "__main__":
    absltest.main()
