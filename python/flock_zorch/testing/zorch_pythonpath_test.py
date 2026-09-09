# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`scripts/zorch_pythonpath.sh`'s memo: it must answer what bazel would, and
it must stop answering when what bazel would say could have changed.

The script is on the harness worker's startup path (`bench_worker_common.sh`
builds PYTHONPATH from it), which is why it caches at all — two bazel client
round-trips per trial, inside the readiness budget. A cache on that path is
also how a worker would silently import the wrong zorch, so the invalidation is
what these cover: every input to bazel's answer has to reach the key, including
the `.bazelrc.user` `--override_module=zorch=...` that `docs/measurement.md`
names as a silent substitution.

Each case runs a COPY of the script over a synthetic checkout, against a bazel
stub, so a run counts invocations instead of timing them and never writes to
the tree it was launched from.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "zorch_pythonpath.sh"

# What the stub reports, mirroring the real pair: an output base and the
# `workspace_root` of a `git_override`d module under it.
WSROOT = "external/zorch+"


class ZorchPythonpathCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="zorch-pythonpath-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        # A synthetic checkout: the script resolves its own root by walking up
        # from its location, so the copy is what makes the key's `pwd` and file
        # contents ours to edit.
        self.checkout = self.tmp / "checkout"
        (self.checkout / "scripts").mkdir(parents=True)
        shutil.copy(SCRIPT, self.checkout / "scripts" / SCRIPT.name)
        self.write_config()

        self.base = self.tmp / "output_base"
        (self.base / WSROOT / "zorch").mkdir(parents=True)
        (self.base / WSROOT / "zorch" / "__init__.py").touch()

        self.calls = self.tmp / "bazel-calls"
        bindir = self.tmp / "bin"
        bindir.mkdir()
        stub = bindir / "bazel"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            f'echo "$@" >> "{self.calls}"\n'
            'case "$1" in\n'
            f'  info) echo "{self.base}" ;;\n'
            f'  cquery) echo "{WSROOT}" ;;\n'
            "esac\n"
        )
        stub.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "FLOCK_ZORCH_JAX_CACHE": str(self.tmp / "cache"),
        }
        self.env.pop("FLOCK_ZORCH_ZORCH_PYTHONPATH_NOCACHE", None)

    def write_config(self, bazelrc_user: str | None = None):
        """The files the key is meant to cover. `bazelrc_user=None` is the
        common case — the file is gitignored and usually absent."""
        (self.checkout / "MODULE.bazel").write_text('git_override(commit = "abc123")\n')
        (self.checkout / "MODULE.bazel.lock").write_text("{}\n")
        (self.checkout / ".bazelrc").write_text(
            "try-import %workspace%/.bazelrc.user\n"
        )
        path = self.checkout / ".bazelrc.user"
        if bazelrc_user is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(bazelrc_user)

    def run_script(self, **env) -> str:
        out = subprocess.run(
            [str(self.checkout / "scripts" / SCRIPT.name)],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()

    def bazel_calls(self) -> int:
        return len(self.calls.read_text().splitlines()) if self.calls.exists() else 0

    def assert_re_resolved(self, before: int):
        """The cold path is `bazel info` plus `bazel cquery`."""
        self.assertEqual(before + 2, self.bazel_calls())

    def test_second_call_answers_from_the_memo_without_asking_bazel(self):
        expected = f"{self.base}/{WSROOT}"
        self.assertEqual(expected, self.run_script())
        self.assert_re_resolved(0)
        first = self.bazel_calls()
        self.assertEqual(expected, self.run_script())
        self.assertEqual(first, self.bazel_calls())

    def test_a_gutted_output_base_is_not_a_hit(self):
        """`bazel clean --expunge` leaves a path that still parses. Importing
        from it fails much later, inside the prover, so the sentinel file — not
        the directory — is what a hit has to prove."""
        self.run_script()
        (self.base / WSROOT / "zorch" / "__init__.py").unlink()
        before = self.bazel_calls()
        self.run_script()
        self.assert_re_resolved(before)

    def test_a_bumped_module_bazel_re_resolves(self):
        """The zorch pin lives in MODULE.bazel; a memo that outlived it would
        hand the worker the previous zorch, and the byte gates are the only
        thing that would notice."""
        self.run_script()
        before = self.bazel_calls()
        (self.checkout / "MODULE.bazel").write_text('git_override(commit = "def456")\n')
        self.run_script()
        self.assert_re_resolved(before)

    def test_adding_a_bazelrc_user_override_re_resolves(self):
        """`.bazelrc` try-imports `.bazelrc.user`, where an
        `--override_module=zorch=...` swaps in a local working copy. It changes
        bazel's answer, so it has to change the key — the substitution
        `docs/measurement.md` warns silently invalidates a measurement."""
        self.run_script()
        before = self.bazel_calls()
        self.write_config(bazelrc_user="common --override_module=zorch=/tmp/zorch\n")
        self.run_script()
        self.assert_re_resolved(before)

    def test_dropping_a_bazelrc_user_override_re_resolves(self):
        """The other direction: going back to the pin is equally a change of
        answer, and an absent file has to key differently from a present one."""
        self.write_config(bazelrc_user="common --override_module=zorch=/tmp/zorch\n")
        self.run_script()
        before = self.bazel_calls()
        self.write_config()
        self.run_script()
        self.assert_re_resolved(before)

    def test_nocache_always_asks(self):
        self.run_script()
        before = self.bazel_calls()
        self.run_script(FLOCK_ZORCH_ZORCH_PYTHONPATH_NOCACHE="1")
        self.assert_re_resolved(before)


if __name__ == "__main__":
    unittest.main()
