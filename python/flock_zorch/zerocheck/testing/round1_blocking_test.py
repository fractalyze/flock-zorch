# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Native unit test for round-1's row partitioning (no golden).

Round-1 is a map-reduce over the row axis: `_extend_rows` transforms along the
LAST axis, so rows are independent, and the reduction collapses them to `[ell]`.
Above `_ROUND1_BLOCK_ROWS` the production path exploits that by weighting rows
from a periodic eq window and applying each block's eq scalar to the partial
sums afterwards. It is admissible because the accumulation is `+` on
`binary_field_ghash`, i.e. XOR — associative and commutative, and the scalar
distributes over it — so regrouping the rows cannot change the result.

"Cannot change the result" is the claim under test, and it is exact rather than
approximate, so the assertion is byte equality at several block counts. The
full-proof byte gate covers the production window against flock; this covers
the regrouping argument itself, cheaply and on CPU.
"""
from __future__ import annotations

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
from absl.testing import absltest, parameterized  # noqa: E402

from flock_zorch import ghash, sumcheck  # noqa: E402
from flock_zorch.testing._util import rand_ghash  # noqa: E402
from flock_zorch.zerocheck import _urm  # noqa: E402

K_SKIP = 6
ELL = 1 << K_SKIP
N_ROWS = 1 << 14


def _inputs(seed: int):
    rng = np.random.default_rng(seed)
    rows = lambda: fnp.asarray(  # noqa: E731
        rng.integers(0, 2, size=(N_ROWS, ELL), dtype=np.uint8)
    )
    a, b, c = rows(), rows(), rows()
    n_challenges = K_SKIP + N_ROWS.bit_length() - 1
    r = rand_ghash(rng, n_challenges)
    return a, b, c, r


def _pack(x):
    """Bit rows -> the packed F128 witness form (uint64 [n/2, 2])."""
    raw = np.packbits(np.asarray(x), axis=None, bitorder="little")
    return fnp.asarray(raw.view(np.uint64).reshape(-1, 2))


def _round1_at(block_rows: int, a, b, c, r, n_partials: int | None = None):
    """`_round1_core` with the block size (and optionally both partials counts)
    forced, as lanes for exact compare."""
    saved = (
        _urm._ROUND1_BLOCK_ROWS,
        _urm._ROUND1_PARTIALS_POINT,
        _urm._ROUND1_PARTIALS_EQX,
    )
    try:
        _urm._ROUND1_BLOCK_ROWS = block_rows
        if n_partials is not None:
            _urm._ROUND1_PARTIALS_POINT = _urm._ROUND1_PARTIALS_EQX = n_partials
        _urm._round1_core.clear_cache()  # the constants are baked into the trace
        return [
            np.asarray(ghash.to_lanes(x)) for x in _urm._round1_core(a, b, c, K_SKIP, r)
        ]
    finally:
        (
            _urm._ROUND1_BLOCK_ROWS,
            _urm._ROUND1_PARTIALS_POINT,
            _urm._ROUND1_PARTIALS_EQX,
        ) = saved
        _urm._round1_core.clear_cache()


def _assert_messages_equal(ref, got, what: str) -> None:
    for name, x, y in zip(("P_AB", "P_C"), ref, got):
        np.testing.assert_array_equal(x, y, err_msg=f"{name} differs: {what}")


class Round1BlockingTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("2_blocks", 1 << 13),
        ("4_blocks", 1 << 12),
        ("16_blocks", 1 << 10),
        ("128_blocks", 1 << 7),
    )
    def test_blocked_matches_unblocked_exactly(self, block_rows: int) -> None:
        a, b, c, r = _inputs(seed=7)
        # A block wider than the input takes the unblocked path — the reference.
        ref = _round1_at(N_ROWS << 1, a, b, c, r)
        got = _round1_at(block_rows, a, b, c, r)
        _assert_messages_equal(ref, got, f"at {N_ROWS // block_rows} blocks")

    @parameterized.named_parameters(("4_blocks", 1 << 12), ("16_blocks", 1 << 10))
    def test_windowed_packed_witness_matches_unblocked_exactly(
        self, block_rows: int
    ) -> None:
        # The windowed path hands the packed F128 witness to the composite
        # whole (no per-block reshape survives), so pin packed x windowed
        # explicitly — the unpacked cases above cannot cover that spelling.
        a, b, c, r = _inputs(seed=23)
        ref = _round1_at(N_ROWS << 1, a, b, c, r)
        got = _round1_at(block_rows, _pack(a), _pack(b), _pack(c), r)
        _assert_messages_equal(ref, got, f"packed at {N_ROWS // block_rows} blocks")

    def test_production_block_size_leaves_small_instances_unblocked(self) -> None:
        # The threshold exists so an instance that already fits keeps its exact
        # program rather than taking the windowed one. m<=28 is below it.
        self.assertGreaterEqual(_urm._ROUND1_BLOCK_ROWS, 1 << 22)


class Round1PartialsCountTest(parameterized.TestCase):
    """`_ROUND1_PARTIALS_POINT` / `_ROUND1_PARTIALS_EQX` split the row axis
    ahead of the same XOR reduce as the blocking above, so no count can change
    the message, on either eq form."""

    @parameterized.named_parameters(("64", 64), ("4096", 4096), ("cap", 1 << 15))
    def test_partials_count_cannot_change_the_result(self, n_partials: int) -> None:
        a, b, c, r = _inputs(seed=19)
        ref = _round1_at(N_ROWS << 1, a, b, c, r)
        got_point = _round1_at(N_ROWS << 1, a, b, c, r, n_partials=n_partials)
        _assert_messages_equal(ref, got_point, f"point form at {n_partials} partials")
        # The blocked lane holds N_ROWS >> 1 rows per block, so a count at or
        # above that clamps back to the production program — coverage the
        # blocking test already owns. Only smaller counts add an eqx-form case.
        if n_partials < N_ROWS >> 1:
            got_eqx = _round1_at(N_ROWS >> 1, a, b, c, r, n_partials=n_partials)
            _assert_messages_equal(ref, got_eqx, f"eqx form at {n_partials} partials")


class Round1CFoldFirstTest(absltest.TestCase):
    """The C track folds first and extends once. Equal to extend-then-reduce by
    linearity of the extend and the φ8 homomorphism — for ANY c rows, no
    identity-C assumption — and exactly, since every reordered sum is XOR.

    The reference below IS the retired per-row C path (extend → φ8 → clmul
    eq-accumulate), rebuilt from the same primitives, so this pins the
    commutation claim itself; the full-proof byte gates pin the wire."""

    def test_fold_first_matches_extend_then_reduce(self) -> None:
        a, b, c, r = _inputs(seed=11)
        eqx = sumcheck.build_eq(r[K_SKIP:])[:, None]
        c_l = _urm._to_u8(_urm._extend_rows(c, K_SKIP))
        phi_c = _urm._PHI_DEV_G[c_l.astype(fnp.int32)]
        ref = np.asarray(ghash.to_lanes(fnp.sum(eqx * phi_c, axis=0)))

        _, got = _urm._round1_core(a, b, c, K_SKIP, r)
        np.testing.assert_array_equal(ref, np.asarray(ghash.to_lanes(got)))

    def test_ab_message_is_untouched_by_the_c_change(self) -> None:
        # P^AB comes out of the same call; pin it against an independent
        # recomputation so a C-side edit can never silently disturb AB.
        a, b, c, r = _inputs(seed=13)
        eqx = sumcheck.build_eq(r[K_SKIP:])[:, None]
        a_l = _urm._extend_rows(a, K_SKIP)
        b_l = _urm._extend_rows(b, K_SKIP)
        ab = _urm._to_u8(a_l * b_l).astype(fnp.int32)
        ref = np.asarray(ghash.to_lanes(fnp.sum(eqx * _urm._PHI_DEV_G[ab], axis=0)))

        got, _ = _urm._round1_core(a, b, c, K_SKIP, r)
        np.testing.assert_array_equal(ref, np.asarray(ghash.to_lanes(got)))

    def test_packed_witness_matches_unpacked_exactly(self) -> None:
        a, b, c, r = _inputs(seed=17)
        ref = [
            np.asarray(ghash.to_lanes(x)) for x in _urm._round1_core(a, b, c, K_SKIP, r)
        ]

        got = [
            np.asarray(ghash.to_lanes(x))
            for x in _urm._round1_core(_pack(a), _pack(b), _pack(c), K_SKIP, r)
        ]
        _assert_messages_equal(ref, got, "packed witness")


if __name__ == "__main__":
    absltest.main()
