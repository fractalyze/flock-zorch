# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`worker_startup_bench`'s parsing and its agreement with the worker it drives.

The bench reads phase marks out of a file `bench_worker` writes, so the two
carry the same list of names in two places. Nothing at runtime notices a
rename: the bench raises `KeyError` on a phase the worker stopped writing, and
silently drops one the worker started writing, in the middle of a measurement
run that costs a permit and several minutes. The first test below is that
guard, read off `bench_worker`'s source so it cannot be satisfied by a stale
copy.

The second concern is that instrumenting startup does not change it. The marks
are gated on `FLOCK_ZORCH_WORKER_TIMELINE`, which the harness clears along with
the rest of the worker's env, and a mark that is not a no-op there would show
up in the score as a slower readiness.
"""

import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path

from flock_zorch.testing import bench_worker
from flock_zorch.testing import worker_startup_bench as bench


class PhaseAgreementTest(unittest.TestCase):
    def test_the_bench_names_exactly_the_phases_the_worker_marks(self):
        """`PHASES` minus its `launch` head is `bench_worker`'s `_mark` calls,
        in order. `proof` is deliberately absent: it lands after the ready file,
        so it is not part of the readiness wall the bench reports."""
        source = Path(bench_worker.__file__).read_text()
        marked = re.findall(r'_mark\("([a-z]+)"\)', source)
        self.assertEqual(list(bench.PHASES[1:]) + ["proof"], marked)

    def test_the_worker_marks_its_first_instant_before_importing_frx(self):
        """`launch` has to be the module's first statement, or the span the
        bench charges to the shim silently absorbs the frx import too."""
        source = Path(bench_worker.__file__).read_text()
        self.assertLess(source.index('_MARKS = [("launch"'), source.index("import frx"))


class InstrumentationIsInertTest(unittest.TestCase):
    def test_marking_without_the_env_var_records_nothing(self):
        """The harness clears the env, so this is the ranked path."""
        self.assertIsNone(bench_worker._TIMELINE)
        before = len(bench_worker._MARKS)
        bench_worker._mark("golden")
        self.assertEqual(before, len(bench_worker._MARKS))


class ParseTest(unittest.TestCase):
    def test_compile_lines_sum_per_program(self):
        """A program is compiled once per distinct shape, and the readiness
        wall pays every one of them, so the per-program row is their sum."""
        stderr = (
            "Finished XLA compilation of jit(_open_jitted) in 5.892991066 sec\n"
            "noise\n"
            "Finished XLA compilation of jit(sample_slice) in 0.085824728 sec\n"
            "Finished XLA compilation of jit(sample_slice) in 0.074462175 sec\n"
        )
        self.assertEqual(
            {"jit(_open_jitted)": 5.892991066, "jit(sample_slice)": 0.160286903},
            {k: round(v, 9) for k, v in bench._parse_compiles(stderr).items()},
        )

    def test_no_compile_lines_is_an_empty_table_not_an_error(self):
        self.assertEqual({}, bench._parse_compiles("nothing to see"))


class ReportTest(unittest.TestCase):
    def _render(self, runs, flag=None) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bench._report(runs, "phase", "readiness", flag)
        return out.getvalue()

    def test_a_key_only_a_later_run_carries_still_gets_a_row(self):
        """Keying off the first run alone drops a program the first worker
        happened not to compile, while the total keeps counting it — the rows
        then do not sum to the number printed under them."""
        table = self._render([{"a": 1.0}, {"a": 1.0, "b": 2.0}])
        self.assertIn("b", [line.split()[0] for line in table.splitlines()[1:]])

    def test_the_total_is_a_wall_some_worker_reached(self):
        """Summing per-key minima across runs invents a faster worker than any
        that ran; the total row is min over per-run sums."""
        table = self._render([{"a": 1.0, "b": 5.0}, {"a": 4.0, "b": 3.0}])
        total = next(l for l in table.splitlines() if l.startswith("readiness"))
        self.assertEqual(["6000.0", "6500.0", "1000.0"], total.split()[1:])

    def test_a_flagged_program_is_marked_and_explained(self):
        """A program with no cache entry recompiled; reporting its seconds as
        deserialization is what any cache-load claim must not do."""
        table = self._render([{"jit(x)": 1.0}], flag={"jit(x)"})
        self.assertIn("jit(x) *", table)
        self.assertIn("no cache entry of this name", table)


class CacheMissTest(unittest.TestCase):
    def test_a_program_without_an_entry_of_its_name_is_a_miss(self):
        """The stem is `jit_<name>`, so `jit(_open_jitted)` looks for
        `jit__open_jitted-<hash>-cache`. Timing cannot make this call."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "jit__open_jitted-0123abc-cache").touch()
            self.assertEqual(
                {"jit(sample_slice)"},
                bench._cache_misses({"jit(_open_jitted)", "jit(sample_slice)"}, d),
            )

    def test_no_cache_dir_flags_nothing_rather_than_everything(self):
        """A worker that reported no cache dir gives no evidence either way;
        marking all 95 programs as misses would be a fabricated finding."""
        self.assertEqual(set(), bench._cache_misses({"jit(x)"}, ""))


if __name__ == "__main__":
    unittest.main()
