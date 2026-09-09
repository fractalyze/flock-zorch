# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`cpu_phase_split`'s classifier, and the promise that instrumenting the prove
does not change the proof.

Both guard mistakes this harness actually made while #324 was measured:

- `classify` keyed on the op name alone, which put `_round1_core`'s 470 ms
  binary-field select + XOR-reduce and `_open_jitted`'s sub-millisecond u32
  proof-of-work reduce — both spelled `select_reduce-window_fusion` — in one
  class. A phase table built on that names the wrong lever.
- `_stem` stripped `jit` rather than `jit_`, so every module rule silently
  missed and 12% of the prove sat in `other` under a table that looked
  complete. A classifier that fails open is worse than one that raises.

The byte guarantee is separate: `prove_bundle(seed, times)` adds barriers and
annotations around the same statements, and a split is only worth reading if
the thing it splits is still the proof the harness scores.
"""

import unittest

import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch.testing import cpu_phase_split as split  # noqa: E402
from flock_zorch.testing._bench_profile import (  # noqa: E402
    PHASES,
    BenchProver,
    constants_golden,
)
from flock_zorch.testing._golden import ART  # noqa: E402


class ClassifyTest(unittest.TestCase):
    def test_same_op_family_splits_by_module(self):
        """The reason the rules are keyed on the pair. `_round1_core`'s is the
        round-1 URM over binary_field_ghash; `_open_jitted`'s is the u32 grind
        reduce, and they differ by two orders of magnitude."""
        op = "select_reduce-window_fusion.1"
        self.assertEqual(split.classify("jit__round1_core", op), "select_xor")
        self.assertEqual(split.classify("jit__open_jitted", op), "grind")

    def test_module_stem_strips_the_jit_prefix_and_the_shape_hash(self):
        """A miss here fails open — every module rule stops matching and the
        residual quietly grows instead of raising."""
        self.assertEqual(split._stem("jit__seg_xor_fold"), "_seg_xor_fold")
        self.assertEqual(split._stem("jit__round1_core-9f3a"), "_round1_core")
        self.assertEqual(split._stem("jit_rs_eq_ind"), "rs_eq_ind")

    def test_module_rule_reaches_a_module_with_no_matching_op_rule(self):
        """`_seg_xor_fold` is all slice/pad/add fusions — no op rule fires, so
        it is the case the module fallback exists for."""
        for op in ("slice_add_fusion.17", "concatenate_pad_fusion", "add_pad_fusion"):
            self.assertEqual(split.classify("jit__seg_xor_fold", op), "xor_fold", op)

    def test_op_families_route_to_their_class(self):
        cases = {
            ("jit__round1_core", "wrapped_ntt.3"): "transform",
            ("jit__commit", "bit_reverse"): "transform",
            ("jit__commit", "blake3.12"): "blake3",
            ("jit_rs_eq_ind", "ffi_call.0"): "ffi",
            ("jit__mlv_sumcheck", "multiply_add_fusion.44"): "ghash_mul",
            ("jit__partial_fold", "select_reduce-window_fusion"): "select_xor",
        }
        for (module, op), want in cases.items():
            self.assertEqual(split.classify(module, op), want, f"{module}/{op}")

    def test_unmatched_lands_in_other_rather_than_a_neighbour(self):
        self.assertEqual(split.classify("jit_sample_scalar", "copy"), "other")

    def test_every_rule_names_a_reported_class(self):
        """A class no report column knows about would vanish from the table
        while still consuming time."""
        named = (
            set(split._MODULE_OP_OVERRIDE.values())
            | {c for _, c in split._OP_CLASS}
            | set(split._MODULE_CLASS.values())
        )
        self.assertLessEqual(named, set(split.KERNEL_CLASSES))

    def test_family_strips_only_the_instance_number(self):
        self.assertEqual(
            split._family("multiply_reduce-window_fusion.46"),
            "multiply_reduce-window_fusion",
        )
        self.assertEqual(split._family("wrapped_ntt"), "wrapped_ntt")
        self.assertEqual(split._family("ffi_call.0"), "ffi_call")


_M22 = constants_golden(22)


@unittest.skipUnless((ART / _M22).exists(), f"{_M22} not dumped in artifacts/")
class InstrumentationTest(unittest.TestCase):
    """The m22 golden is enough: this asserts the shape of the instrumentation,
    not a number. Timings at this size are dominated by fixed floors and are
    not a measurement — see `docs/measurement.md`."""

    @classmethod
    def setUpClass(cls):
        from flock_zorch.testing.blake3_ligerito_oracle_test import load

        cls.bp = BenchProver(load(_M22))

    def test_timing_does_not_change_the_proof(self):
        """The barriers and annotations must be transparent — otherwise the
        split describes a prove the harness never runs."""
        plain = self.bp.prove_bundle(7)
        timed = self.bp.prove_bundle(7, {})
        self.assertEqual(plain, timed)

    def test_every_phase_is_recorded_and_accounts_for_the_prove(self):
        """`sum` vs the caller's own wall is the self-check that no statement
        escaped a phase; a gap means work billed to nobody."""
        import time

        times: dict[str, float] = {}
        t0 = time.perf_counter()
        self.bp.prove_bundle(7, times)
        wall = (time.perf_counter() - t0) * 1e3
        self.assertEqual(set(times), set(PHASES))
        self.assertAlmostEqual(sum(times.values()), wall, delta=0.10 * wall)


if __name__ == "__main__":
    unittest.main()
