# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`rival_compare` turns another repo's stdout into a cross-prover ratio.

Everything here guards the same failure: a number that is wrong but looks fine.
The rival's output format is not ours to control, the asymmetry corrections are
subtractions nobody re-derives once they are in a table, and the instance gate is
the only thing standing between "2.3x" and a ratio between two different
problems. All of it is pure, so it gates on CPU.
"""

import unittest

from flock_zorch.testing import rival_compare as rc

# A verbatim `bench_ligerito fast32` capture (RTX 5090, 2026-08-22), trimmed to
# the lines the parser reads. Kept verbatim rather than synthesized: the point of
# the test is that we can still read what that binary actually prints.
RIVAL_STDOUT = """Device: NVIDIA GeForce RTX 5090 | 170 SMs
Ligerito open [m32_fast config, grinding OFF]: log_n=25 initial_k=6 r=5 k_rec=3 \
rates=1..6  queries=218,106,71,53,43,36  ood=0,1,1,1,1,1
  [l0-detail] ntt(fused rate-extend, 2^25 x 64 lanes) 3.180 | merkle(2^20 leaves \
x 1024 B) 4.340 ms
  open 3.48 ms | commit 0.42  fold 0.91  ood 0.13  multiproof 1.44  induce 0.38 \
 introduce/glue 0.20
  resident chain: witness-gen 1.11  l0-commit 7.52  zerocheck 13.55  lincheck \
0.76  eq-build 0.24 ms
  device overhead: cudaMalloc 0.31  cudaFree 0.12  H2D 0.44 ms | bench-only \
input fill 0.69 ms
  >>> prove wall 27.35 ms (26.66 excl. bench fill) | phase total 27.34 ms | \
unattributed 0.01 ms (+0.02%)
"""

# The m32 golden's Ligerito config, as `read_ligerito_config` returns it.
OURS_M32 = {
    "log_n": 25,
    "initial_k": 6,
    "recursive_ks": [3, 3, 3, 3, 3],
    "queries": [218, 106, 71, 53, 43, 36],
    "fold_grinding_bits": [19, 14, 11, 8, 6, 4],
}


class ParseRivalTest(unittest.TestCase):
    def test_reads_every_field_off_a_real_capture(self):
        r = rc.parse_rival(RIVAL_STDOUT)
        self.assertEqual(r.wall_ms, 27.35)
        self.assertEqual(r.excl_fill_ms, 26.66)
        self.assertEqual(r.witness_ms, 1.11)
        self.assertEqual(r.unattributed_pct, 0.02)
        self.assertEqual(
            r.phases,
            {"commit": 7.52, "zerocheck": 13.55, "lincheck": 0.76, "open": 3.48},
        )
        self.assertEqual(r.queries, (218, 106, 71, 53, 43, 36))
        self.assertEqual(
            (r.log_n, r.initial_k, r.recursive_steps, r.k_rec), (25, 6, 5, 3)
        )

    def test_open_is_the_phase_total_not_the_multiproof_substep(self):
        """Their `open ... ms |` line leads with the whole open phase and then
        itemizes it, and one of the items is *also* called `open` in the source
        (`ph.open` is the multiproof step). Taking the wrong one understates the
        phase we are furthest behind on by ~2x."""
        self.assertEqual(rc.parse_rival(RIVAL_STDOUT).phases["open"], 3.48)

    def test_a_missing_line_raises_rather_than_defaulting(self):
        """The guarded failure is silent, not loud: if an upstream reformat drops
        a line and the parser fills 0.0, the ratio moves and nothing looks
        broken."""
        for drop in ("resident chain", ">>> prove wall", "log_n="):
            trimmed = "\n".join(
                ln for ln in RIVAL_STDOUT.splitlines() if drop not in ln
            )
            with self.assertRaises(rc.ParseError):
                rc.parse_rival(trimmed)

    def test_comparable_ms_removes_what_their_harness_reports(self):
        """Their wall counts a bench-only input fill and witness generation; our
        phase split counts neither. Both are figures their own output carries, so
        the correction stays a measurement."""
        r = rc.parse_rival(RIVAL_STDOUT)
        self.assertAlmostEqual(r.comparable_ms, 26.66 - 1.11, places=6)


class InstanceGateTest(unittest.TestCase):
    def test_the_real_pair_matches(self):
        """`m32` and `fast32` are just labels. This is the check that makes the
        comparison legitimate, so it has to pass on the actual pair."""
        self.assertEqual(
            rc.instance_mismatches(OURS_M32, rc.parse_rival(RIVAL_STDOUT)), []
        )

    def test_each_constant_is_actually_compared(self):
        for key, bad in (
            ("log_n", 26),
            ("initial_k", 5),
            ("queries", [218, 106, 71, 53, 43, 35]),
        ):
            with self.subTest(key=key):
                mismatches = rc.instance_mismatches(
                    {**OURS_M32, key: bad}, rc.parse_rival(RIVAL_STDOUT)
                )
                self.assertTrue(any(m.startswith(key) for m in mismatches), mismatches)

    def test_a_differing_recursion_depth_is_caught(self):
        mismatches = rc.instance_mismatches(
            {**OURS_M32, "recursive_ks": [3, 3, 3]}, rc.parse_rival(RIVAL_STDOUT)
        )
        self.assertTrue(any(m.startswith("recursive_steps") for m in mismatches))

    def test_a_nonuniform_k_rec_is_reported_not_silently_accepted(self):
        """Their log prints one `k_rec`, so a config whose recursive ks differ
        cannot be compared against it field-for-field — say so rather than
        checking element 0 and moving on."""
        mismatches = rc.instance_mismatches(
            {**OURS_M32, "recursive_ks": [3, 4, 3, 3, 3]}, rc.parse_rival(RIVAL_STDOUT)
        )
        self.assertTrue(any("varies" in m for m in mismatches), mismatches)


class ContentionTest(unittest.TestCase):
    """An OOM on this box is contention until the card is proven idle.

    The measurement this protects is a cross-prover ratio, and a contended arm
    does not read as merely noisy — it produces an OOM, an inflated wall and a
    huge spread at once, all of which look like properties of the prover under
    test rather than of the neighbour that took the card.
    """

    def test_each_layer_that_reports_an_oom_is_recognized(self):
        """Three different layers surface it, each with its own wording: frx's
        allocator, XLA's autotuner, and the rival's raw driver call."""
        for text in (
            "EXECUTION FAILED: RESOURCE_EXHAUSTED: Out of memory while trying",
            "Failed to allocate 1.20GiB: CUDA_ERROR_OUT_OF_MEMORY: out of memory",
            "CUDA err out of memory: cudaMallocAsync 1.00 GiB, device free 0.01",
        ):
            with self.subTest(text=text[:40]):
                self.assertIsNotNone(rc._oom_signature(text))

    def test_a_clean_run_is_not_flagged(self):
        self.assertIsNone(rc._oom_signature(RIVAL_STDOUT))


class FoldGrindCensusTest(unittest.TestCase):
    """The fold PoW schedule is the asymmetry that reframed this comparison:
    flock's bench performs no fold grinds at all, ours performs 21."""

    def setUp(self):
        from flock_zorch.testing import prove_phase_bench

        self.bench = prove_phase_bench

    def test_m32_schedule_matches_the_choreography(self):
        """Mirrors `FlockChoreography.fold_grind_bits`: level `l` round `j`
        grinds `bits[l] - j`, only when > 0. At m32 that is 19+18+...+14 on
        level 0 and three rounds on each of five recursive levels."""
        cost = self.bench.fold_grind_census(OURS_M32)
        self.assertEqual(cost.grinds, 21)
        expected = sum(
            1 << b
            for level, k in enumerate((6, 3, 3, 3, 3, 3))
            for b in (OURS_M32["fold_grinding_bits"][level] - j for j in range(k))
            if b > 0
        )
        self.assertEqual(cost.expected_attempts, expected)
        self.assertEqual(cost.expected_attempts, 1_065_036)

    def test_the_grind_window_floors_every_search(self):
        """`grind_search` tests a whole `GRIND_WINDOW` batch per `while_loop`
        step, so an easy grind costs the same as a 16-bit one. That nearly
        doubles the real total, and it is the difference between "the fold PoW
        is level 0" and "level 0 is half of it"."""
        from zorch.grind import GRIND_WINDOW

        cost = self.bench.fold_grind_census(OURS_M32)
        self.assertEqual(cost.windowed_hashes, 1 << 21)
        self.assertGreater(cost.windowed_hashes, 1.9 * cost.expected_attempts)
        # 18 of the 21 sit at <= 16 bits and each still pays a full window;
        # only the top three (19, 18, 17) exceed it.
        cheap = sum(
            1
            for level, k in enumerate((6, 3, 3, 3, 3, 3))
            for b in (OURS_M32["fold_grinding_bits"][level] - j for j in range(k))
            if 0 < b <= GRIND_WINDOW.bit_length() - 1
        )
        self.assertEqual(cheap, 18)

    def test_level_zero_dominates_the_attempts_but_not_the_hashes(self):
        """The correction this encodes: level 0 is 97% of the *expected
        attempts* and only 53% of the *hashes actually evaluated*. Scoping a fix
        off the first number would aim at the wrong 15 grinds."""
        total = self.bench.fold_grind_census(OURS_M32)
        l0 = self.bench.fold_grind_census(
            {**OURS_M32, "recursive_ks": [], "fold_grinding_bits": [19]}
        )
        self.assertGreater(l0.expected_attempts / total.expected_attempts, 0.96)
        self.assertLess(l0.windowed_hashes / total.windowed_hashes, 0.60)

    def test_rounds_that_taper_to_zero_do_not_grind(self):
        cfg = {"initial_k": 6, "recursive_ks": [], "fold_grinding_bits": [2]}
        cost = self.bench.fold_grind_census(cfg, window=1)
        self.assertEqual(
            (cost.grinds, cost.expected_attempts), (2, (1 << 2) + (1 << 1))
        )

    def test_drop_zeroes_the_schedule_and_reports_what_it_removed(self):
        cfg = dict(OURS_M32)
        self.assertEqual(self.bench.drop_fold_grinds(cfg), (21, 1_065_036, 1 << 21))
        self.assertEqual(cfg["fold_grinding_bits"], [0] * 6)
        self.assertEqual(self.bench.fold_grind_census(cfg), (0, 0, 0))

    def test_the_choreography_actually_stops_grinding(self):
        """The census counts what the config *says*; this checks what the
        prover would *do*. A flag whose structure did not move is an unfired
        experiment, not a null result — so the arm is only meaningful if
        `FlockChoreography` really emits no fold grind after the drop.
        """
        from flock_zorch.pcs.ligerito import FlockChoreography

        def emitted(bits):
            chor = FlockChoreography(
                fold_grinding_bits=tuple(bits),
                query_grinding_bits=(0,) * len(bits),
            )
            return [
                (level, j)
                for level, k in enumerate((6, 3, 3, 3, 3, 3))
                for j in range(k)
                if chor.fold_grind_bits(level, j) is not None
            ]

        cfg = dict(OURS_M32)
        self.assertEqual(len(emitted(cfg["fold_grinding_bits"])), 21)
        self.bench.drop_fold_grinds(cfg)
        self.assertEqual(emitted(cfg["fold_grinding_bits"]), [])

    def test_drop_leaves_the_query_grinds_alone(self):
        """0-bit query grinds still put a nonce on the wire, and flock's bench
        does exactly that. Zeroing them too would make the arms differ again, in
        the other direction."""
        cfg = {**OURS_M32, "grinding_bits": [0, 0, 0, 0, 0, 0]}
        self.bench.drop_fold_grinds(cfg)
        self.assertEqual(cfg["grinding_bits"], [0, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
