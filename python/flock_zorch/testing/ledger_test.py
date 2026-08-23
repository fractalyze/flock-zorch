# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The check-run wire contract and the conclusion semantics.

`output.summary` is the only thing a consumer sees, and the work-map view
parses it by looking for `LEDGER_MARK` and reading the fenced JSON after it.
Nothing in CI exercises that consumer, so the round trip is pinned here: a
record that leaves `summarise` must come back byte-identical, or a trajectory
silently starts mixing schemas.

The conclusion mapping is pinned for the same reason. `neutral` is what keeps
a busy GPU from reading as a regression, and it is only correct if a refusal
also leaves the exit code green.
"""

from __future__ import annotations

import json
import re
from typing import Any

from absl.testing import absltest

from flock_zorch.testing._ledger import (
    LEDGER_MARK,
    Outcome,
    comparability,
    publish,
    summarise,
    summarise_outcome,
)

# How the consumer finds the block (`fetch_measurements.py` uses this regex).
_BLOCK_RE = re.compile(re.escape(LEDGER_MARK) + r"\s*```json\s*(.*?)```", re.S)


def _fingerprint(**over: Any) -> dict[str, Any]:
    base = {
        "toolchain": {"ptxas": "13.3", "nvlink": "13.3"},
        "device": {
            "name": "NVIDIA GeForce RTX 5090",
            "driver": "580.126.09",
            "pinned_to": "0",
        },
        "pins": {
            "installed": {"frx": "0.10.2.dev20260821142028"},
            "zorch_commit": "b3003e7cb234abede44b8b3ab46c5269e8165f37",
            "lockstep": True,
        },
        "overrides": {},
        "runtime": {"XLA_PYTHON_CLIENT_ALLOCATOR": None},
        "source": {"sha": "a" * 40},
    }
    base.update(over)
    return base


def _record(**over: Any) -> dict[str, Any]:
    return {
        "benchmarks": [
            {
                "suite": "prove_phase_bench",
                "name": "blake3_m32",
                "variant": "throughput",
                "metrics": {"throughput": 4690000.0, "latency": 55.95},
                "instance": {"m": 32, "hashes": 262144},
                "env": _fingerprint(**over),
            }
        ],
        "window": {"mode": "throughput", "runs": 10, "processes": 1},
    }


class WireContractTest(absltest.TestCase):
    def test_record_survives_the_round_trip(self) -> None:
        record = _record()
        _, summary = summarise(record, [], "m32")
        block = _BLOCK_RE.search(summary)
        assert block is not None, "the consumer's marker+fence is not in the summary"
        self.assertEqual(json.loads(block.group(1)), record)

    def test_summary_fits_the_field(self) -> None:
        # `output.summary` is capped at 65535 characters; a truncated block is
        # unparseable, so the record must stay far under it.
        _, summary = summarise(_record(), [], "m32")
        self.assertLess(len(summary), 65535)

    def test_headline_carries_the_number(self) -> None:
        title, _ = summarise(_record(), [], "m32")
        self.assertIn("4.69M", title)
        self.assertIn("55.95", title)
        self.assertIn("m32", title)

    def test_headline_scales_to_a_slow_arm(self) -> None:
        # A CPU arm lands three orders of magnitude down; fixed millions would
        # render 701/s as "0.00M", which reads as a broken run, not a slow one.
        record = _record()
        record["benchmarks"][0]["metrics"] = {"throughput": 701.0, "latency": 365.04}
        title, _ = summarise(record, [], "m22")
        self.assertIn("701", title)
        self.assertNotIn("0.00M", title)

    def test_incomparable_is_stated_not_implied(self) -> None:
        _, summary = summarise(_record(), ["toolchain.ptxas: '13.3' -> '12.9'"], "m32")
        self.assertIn("NOT comparable", summary)
        self.assertIn("12.9", summary)

    def test_an_override_off_the_pin_is_named(self) -> None:
        # The #200 erratum in one line: an override is not the problem, an
        # override that is not the declared pin is.
        record = _record(overrides={"zorch": {"head": "d" * 40, "matches_pin": False}})
        _, summary = summarise(record, [], "m32")
        self.assertIn("NOT the declared pin", summary)

    def test_absent_source_pin_is_omitted_not_rendered_as_none(self) -> None:
        # A repo that pins no source dep this way should not grow a "None" row.
        record = _record(pins={"installed": {}, "zorch_commit": None, "lockstep": True})
        _, summary = summarise(record, [], "m32")
        self.assertNotIn("zorch pin", summary)


class ComparabilityTest(absltest.TestCase):
    def test_environment_only_check_ignores_the_baseline_window(self) -> None:
        # The pre-flight check runs before a window exists. Diffing against the
        # baseline's window would report every such check as drifted.
        baseline = {**_fingerprint(), "window": {"mode": "throughput", "runs": 10}}
        self.assertEqual(comparability(baseline, _fingerprint()), [])

    def test_window_change_is_caught_once_it_is_known(self) -> None:
        baseline = {**_fingerprint(), "window": {"mode": "throughput", "runs": 10}}
        found = comparability(
            baseline, _fingerprint(), {"mode": "barriered", "runs": 10}
        )
        self.assertTrue(found)

    def test_no_baseline_is_not_a_verdict(self) -> None:
        self.assertEqual(comparability(None, _fingerprint()), [])


class ConclusionTest(absltest.TestCase):
    """A busy GPU must not turn a branch red; a broken bench must."""

    def test_a_refusal_is_green(self) -> None:
        code = publish(
            Outcome(refusal="1 other compute process on the card"),
            name="bench (blake3 m32)",
            head_sha="a" * 40,
            subject="m32",
        )
        self.assertEqual(code, 0)

    def test_a_broken_bench_is_red(self) -> None:
        code = publish(
            Outcome(error="ImportError"),
            name="bench (blake3 m32)",
            head_sha="a" * 40,
            subject="m32",
        )
        self.assertEqual(code, 1)

    def test_a_measurement_is_green(self) -> None:
        code = publish(
            Outcome(record=_record()),
            name="bench (blake3 m32)",
            head_sha="a" * 40,
            subject="m32",
        )
        self.assertEqual(code, 0)

    def test_a_refusal_says_no_point_was_added(self) -> None:
        title, summary = summarise_outcome(Outcome(refusal="contended card"))
        self.assertEqual(title, "not measured")
        self.assertIn("not that it got slower", summary)
        self.assertIn("contended card", summary)

    def test_a_failure_does_not_claim_it_was_merely_unmeasured(self) -> None:
        title, summary = summarise_outcome(Outcome(error="boom"))
        self.assertEqual(title, "the bench failed")
        self.assertNotIn("not that it got slower", summary)


if __name__ == "__main__":
    absltest.main()
