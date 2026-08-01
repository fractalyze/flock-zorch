"""Native tests for the primitive census's evidence classification."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from flock_zorch.testing import primitive_census as census


def _point(ns: float, *, roof: float | None = None, lane: float | None = None):
    return census.Point(
        log_elements=10,
        elements=1024,
        milliseconds=ns * 1024 / 1e6,
        ns_per_element=ns,
        billion_elements_per_second=1 / ns,
        effective_gbps=1,
        roofline_percent=roof,
        lane_milliseconds=(ns * 1024 / 1e6 / lane if lane else None),
        native_over_lane=lane,
    )


class PrimitiveCensusTest(unittest.TestCase):
    def test_classifies_dispatch_and_steady_state_defect_independently(self):
        self.assertEqual(
            census.classify([_point(20), _point(2, roof=20)], 2, 2),
            "lowering-defect+dispatch-bound",
        )
        self.assertEqual(
            census.classify([_point(20), _point(2, roof=80)], 2, 2),
            "dispatch-bound",
        )

    def test_lane_equivalent_takes_precedence_over_memory_roofline(self):
        # Random gather cannot reach sequential DRAM peak, but matching the
        # identical uint64 access proves the field dtype is not the cause.
        self.assertEqual(
            census.classify([_point(5), _point(4, roof=10, lane=1.0)], 2, 2),
            "efficient-or-inconclusive",
        )
        self.assertEqual(
            census.classify([_point(5), _point(4, roof=80, lane=3.0)], 2, 2),
            "lowering-defect",
        )

    def test_clmul_roofline_requires_issue_peak(self):
        roofline = census.Roofline(bytes_per_element=48, clmul_per_element=1)
        self.assertIsNone(census.roofline_ns_per_element(roofline, 1000, None))
        self.assertAlmostEqual(
            census.roofline_ns_per_element(roofline, 1000, 10), 0.1
        )

    def test_ntt_roofline_counts_fused_passes(self):
        self.assertEqual(census._ntt_roofline(1 << 24), census.Roofline(96, 96))
        extend = next(
            operation
            for operation in census.OPERATIONS
            if operation.name == "additive_ntt_extend"
        )
        self.assertEqual(extend.roofline(1 << 24), census.Roofline(64, 48))

    def test_op_counts_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.json"
            path.write_text(json.dumps({"add": 12}))
            self.assertEqual(census._load_counts(path), {"add": 12})
            path.write_text(json.dumps({"add": -1}))
            with self.assertRaises(ValueError):
                census._load_counts(path)
            path.write_text(json.dumps({"typo": 1}))
            with self.assertRaises(ValueError):
                census._load_counts(path)


if __name__ == "__main__":
    unittest.main()
