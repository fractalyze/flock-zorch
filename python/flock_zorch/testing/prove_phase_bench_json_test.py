# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Native unit test for `prove_phase_bench.py`'s `--json` report shape.

The autoresearch platform ingests this file's JSON, so its shape is a wire
contract like a proof's — get a key or a variant wrong and the platform either
drops the row silently or misreads a diagnostic-mode number as a throughput
one. `_throughput_rows` / `_phase_rows` / `build_report` are pure functions of
numbers a real prove would have already produced, so this exercises the
contract from fabricated numbers, no GPU and no golden required. The one
non-pure surface, `env_provenance`, is checked only for its shape and for
never raising off a probe that a CPU box legitimately lacks (no `nvidia-smi`,
no `ptxas`) — its values are environment-dependent by design.

Pure host: this is dict/JSON shaping, not field math.
"""
from __future__ import annotations

import json

from absl.testing import absltest

from flock_zorch.testing import prove_phase_bench as ppb
from flock_zorch.testing._util import json_row


class JsonRowTest(absltest.TestCase):
    def test_drops_none_valued_metrics(self):
        row = json_row(
            "prove_phase_bench", "blake3_m24", "throughput", memory=None, throughput=1.0
        )
        self.assertEqual(
            row,
            {
                "suite": "prove_phase_bench",
                "name": "blake3_m24",
                "variant": "throughput",
                "metrics": {"throughput": 1.0},
            },
        )

    def test_keeps_zero_valued_metrics(self):
        """`0` is a real measurement (e.g. a zero-byte peak); only `None`, the
        marker for "this backend never reported one", is dropped."""
        row = json_row("prove_phase_bench", "blake3_m24", "commit", memory=0)
        self.assertEqual(row["metrics"], {"memory": 0})


class ThroughputRowsTest(absltest.TestCase):
    def test_shape_and_values(self):
        # n_hash=1000 hashes in wall_ms=200ms -> 5000 hash/s; compile_ms=50ms
        # warmup, peak=1234 bytes.
        rows = ppb._throughput_rows("blake3_m24", 1000, 200.0, 50.0, 1234)
        self.assertLen(rows, 1)
        row = rows[0]
        self.assertEqual(row["suite"], "prove_phase_bench")
        self.assertEqual(row["name"], "blake3_m24")
        self.assertEqual(row["variant"], "throughput")
        self.assertEqual(
            row["metrics"],
            {
                "throughput": 5000.0,
                "latency": 0.2,
                "compile_time": 0.05,
                "memory": 1234,
            },
        )

    def test_missing_compile_time_and_memory_are_omitted_not_null(self):
        row = ppb._throughput_rows("blake3_m24", 1000, 200.0, None, None)[0]
        self.assertEqual(row["metrics"], {"throughput": 5000.0, "latency": 0.2})

    def test_json_roundtrips(self):
        """The platform reads this off disk, not off a live Python object —
        assert it survives an actual dump/load, not just equality in memory."""
        rows = ppb._throughput_rows("blake3_m24", 1000, 200.0, 50.0, 1234)
        self.assertEqual(json.loads(json.dumps(rows)), rows)


class PhaseRowsTest(absltest.TestCase):
    def test_one_row_per_phase_no_throughput_metric(self):
        parts = {"commit": 1.0, "zerocheck": 2.0, "lincheck": 3.0, "open": 4.0}
        rows = ppb._phase_rows("blake3_m24", parts, 50.0, 1234)
        self.assertEqual([r["variant"] for r in rows], list(ppb.PHASES))
        for row, phase in zip(rows, ppb.PHASES):
            self.assertEqual(row["suite"], "prove_phase_bench")
            self.assertEqual(row["name"], "blake3_m24")
            # Diagnostic-mode wall time is a documented over-estimate
            # (docs/measurement.md: ~14% slower than --throughput), so no
            # variant produced here may carry a "throughput" metric key —
            # that key is reserved for _throughput_rows's own row.
            self.assertNotIn("throughput", row["metrics"])
            self.assertEqual(row["metrics"]["latency"], parts[phase] / 1e3)
            self.assertEqual(row["metrics"]["compile_time"], 0.05)
            self.assertEqual(row["metrics"]["memory"], 1234)

    def test_missing_memory_omitted_across_every_phase_row(self):
        parts = {"commit": 1.0, "zerocheck": 2.0, "lincheck": 3.0, "open": 4.0}
        rows = ppb._phase_rows("blake3_m24", parts, 50.0, None)
        for row in rows:
            self.assertNotIn("memory", row["metrics"])


class BuildReportTest(absltest.TestCase):
    def test_wraps_rows_and_stamps_the_same_env_on_each(self):
        rows = [
            {
                "suite": "prove_phase_bench",
                "name": "a",
                "variant": "throughput",
                "metrics": {},
            },
            {
                "suite": "prove_phase_bench",
                "name": "b",
                "variant": "commit",
                "metrics": {},
            },
        ]
        env = {"ptxas": "release 13.3", "driver": "580.1", "allocator": "default"}
        report = ppb.build_report(rows, env)
        self.assertEqual(list(report), ["benchmarks"])
        self.assertLen(report["benchmarks"], 2)
        for row in report["benchmarks"]:
            self.assertEqual(row["env"], env)
        # Each row's env is its own copy: mutating one must not mutate another.
        report["benchmarks"][0]["env"]["ptxas"] = "mutated"
        self.assertEqual(report["benchmarks"][1]["env"]["ptxas"], "release 13.3")

    def test_json_serializable_end_to_end(self):
        rows = ppb._throughput_rows("blake3_m24", 1000, 200.0, 50.0, 1234)
        report = ppb.build_report(
            rows, {"ptxas": None, "driver": None, "allocator": "default"}
        )
        self.assertEqual(
            json.loads(json.dumps(report))["benchmarks"][0]["name"], "blake3_m24"
        )


class EnvProvenanceTest(absltest.TestCase):
    def test_never_raises_and_has_the_expected_keys(self):
        """No GPU, no ptxas, and no nvidia-smi are guaranteed here (a CPU CI
        box has none of the three) — this is the environment this test must
        pass in, not one it mocks."""
        env = ppb.env_provenance()
        self.assertEqual(set(env), {"ptxas", "driver", "allocator"})
        for key in ("ptxas", "driver"):
            self.assertTrue(env[key] is None or isinstance(env[key], str))
        self.assertIsInstance(env["allocator"], str)


class PeakMemoryBytesTest(absltest.TestCase):
    def test_none_on_a_backend_without_memory_stats(self):
        """The CPU backend this test runs on (the whole point is not needing a
        GPU) doesn't implement `memory_stats`, so this exercises exactly the
        fallback branch a GPU run's device never hits."""
        self.assertIsNone(ppb._peak_memory_bytes())


if __name__ == "__main__":
    absltest.main()
