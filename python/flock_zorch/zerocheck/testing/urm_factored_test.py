# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-equality of the portable factored-eq round-1 core against the composite.

`_round1_core_factored` folds the protocol's 7 pinned inner dims on gf8 bytes —
small by shift-reduce, medium by the convert table — where `_round1_core`
lifts every product bit to F128 first. Same message, two formulations; the
claim under test is that "same" is exact, so the assertion is byte equality on
both wire fields.

The full-proof byte gate covers the production dispatch against flock; this
covers the factorization itself, cheaply and on CPU. It also pins the guard:
off the pinned inner challenges the factorization does not hold, and the
dispatcher must fall back, and it exhaustively checks `_gf8_reduce`, which
the Triton kernel shares — the one symbol here a CPU box cannot reach through
its own consumer.
"""
from __future__ import annotations

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
from absl.testing import absltest, parameterized  # noqa: E402

from flock_zorch import ghash  # noqa: E402
from flock_zorch.testing._util import rand_ghash  # noqa: E402
from flock_zorch.zerocheck import _urm  # noqa: E402
from flock_zorch.zerocheck import prover as zc_prover  # noqa: E402

K_SKIP = 6
ELL = 1 << K_SKIP


def _protocol_r(rng, m: int):
    """The zerocheck challenge vector `prover.prove` builds: k_skip skip
    challenges, the pinned inner 7, then the outer ones."""
    n_outer = m - K_SKIP - zc_prover.N_INNER
    return fnp.concatenate(
        [
            rand_ghash(rng, K_SKIP),
            zc_prover._SMALL_G,
            zc_prover._MEDIUM_G,
            rand_ghash(rng, n_outer),
        ]
    )


def _packed(rng, m: int):
    """A random packed F128 witness (uint64 [2^(m-7), 2])."""
    return fnp.asarray(rng.integers(0, 2**64, size=(1 << (m - 7), 2), dtype=np.uint64))


class UrmFactoredTest(parameterized.TestCase):
    @parameterized.parameters(13, 15, 17)
    def test_matches_composite(self, m: int) -> None:
        rng = np.random.default_rng(m)
        a, b, c = _packed(rng, m), _packed(rng, m), _packed(rng, m)
        r = _protocol_r(rng, m)

        want = _urm._round1_core(a, b, c, K_SKIP, r)
        got = _urm._round1_core_factored(a, b, c, K_SKIP, r)

        for name, w, g in zip(("P^AB", "P^C"), want, got):
            np.testing.assert_array_equal(
                np.asarray(ghash.to_lanes(g)),
                np.asarray(ghash.to_lanes(w)),
                err_msg=f"{name} diverges at m={m}",
            )

    def test_matches_composite_on_uncoerced_c_bytes(self) -> None:
        """The composite reads C as `!= 0`, and `_round1_input_rows` hands some
        input forms straight through as raw bytes — so a C byte above 1 must
        fold to the same message, not index off its convert-table row."""
        m = 15
        n_rows = 1 << (m - K_SKIP)
        rng = np.random.default_rng(23)
        flat = lambda hi: fnp.asarray(  # noqa: E731
            rng.integers(0, hi, size=n_rows * ELL, dtype=np.uint8)
        )
        a, b, c = flat(2), flat(2), flat(256)
        r = _protocol_r(rng, m)

        want = _urm._round1_core(a, b, c, K_SKIP, r)
        got = _urm._round1_core_factored(a, b, c, K_SKIP, r)
        for name, w, g in zip(("P^AB", "P^C"), want, got):
            np.testing.assert_array_equal(
                np.asarray(ghash.to_lanes(g)),
                np.asarray(ghash.to_lanes(w)),
                err_msg=f"{name} diverges on raw C bytes",
            )

    def test_dispatch_takes_the_factored_core_on_cpu(self) -> None:
        """The pinned inner challenges select it; anything else falls back."""
        if frx.default_backend() != "cpu":
            self.skipTest("the factored core is the CPU tier's formulation")
        m = 15
        rng = np.random.default_rng(3)
        r = _protocol_r(rng, m)
        self.assertTrue(_urm._round1_factored_ok(m, K_SKIP, r))

        unpinned = fnp.concatenate([r[:K_SKIP], rand_ghash(rng, m - K_SKIP)])
        self.assertFalse(_urm._round1_factored_ok(m, K_SKIP, unpinned))

    def test_dispatch_falls_back_under_a_trace(self) -> None:
        """A caller that jits the whole prove hands the guard a tracer, whose
        values it cannot read — so it must decline rather than raise."""
        m = 15
        r = _protocol_r(np.random.default_rng(5), m)
        ok = frx.jit(lambda rr: _urm._round1_factored_ok(m, K_SKIP, rr))(r)
        self.assertFalse(ok)

    @parameterized.parameters(np.uint16, np.int32)
    def test_gf8_reduce_is_exhaustively_the_aes_poly_remainder(self, dtype) -> None:
        """Every input the shift-reduce can produce, against an independent
        long division by 0x11B.

        `_urm_pallas` calls this on int32 lanes and `_fold_small` on uint16, so
        both widths are checked. The Triton kernel is the other consumer and no
        CPU box can compile it, which is what makes exhaustive coverage here
        worth its cost — it is 2¹⁵ values."""
        p = np.arange(1 << 15, dtype=dtype)
        want = p.copy()
        for bit in range(14, 7, -1):
            want ^= ((want >> bit) & 1) * dtype(0x11B << (bit - 8))
        got = np.asarray(_urm._gf8_reduce(fnp.asarray(p)))
        np.testing.assert_array_equal(got, want)

    def test_convert_table_is_the_gamma_scaled_phi8_lift(self) -> None:
        """`convert[b][v] == γᵇ · φ₈(v)`, checked against the ghash multiply
        rather than against the lane doubling that builds it."""
        table = ghash.to_ghash(fnp.asarray(_urm._build_convert_table().reshape(-1, 2)))
        gamma = ghash.to_ghash(fnp.asarray(np.array([[2, 0]], dtype=np.uint64)))[0]
        want = _urm._PHI_DEV_G
        for b in range(_urm._MEDIUM):
            np.testing.assert_array_equal(
                np.asarray(ghash.to_lanes(table[b * 256 : (b + 1) * 256])),
                np.asarray(ghash.to_lanes(want)),
                err_msg=f"convert row {b}",
            )
            want = want * gamma


if __name__ == "__main__":
    absltest.main()
