# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Python-native gate (no golden): the CPU byte-table fold is `rs_eq_ind`.

`_rs_eq_ind_cpu` exists only to move that fold off zorch's single-threaded host
FFI handler, so the one thing it owes is the handler's bytes. A divergence
would flip every Fiat-Shamir draw after the open, which the proof gates do
catch — but only as "the proof diverges", naming nothing. Compared here against
the zorch kernel directly, and against a bit-loop transcription of flock's
`ring_switch::fold_b128_elems_naive` so a shared misreading of the byte order
cannot pass both.

Sizes span the two shapes the fold meets: a `2^L` suffix tensor (the ring-switch
claims) and the odd lengths a padded or degenerate claim can hand it.
"""

import numpy as np
from absl.testing import absltest, parameterized
from zorch.pcs.ring_switch import rs_eq_ind

from flock_zorch import ghash
from flock_zorch.pcs.ring_switch import LOG_PACKING, _rs_eq_ind_cpu
from flock_zorch.testing._util import rand_ghash

_WIDTH = 1 << LOG_PACKING  # 128


def _fold_naive(tensor_lanes: np.ndarray, eq_lanes: np.ndarray) -> np.ndarray:
    """flock `ring_switch::fold_b128_elems_naive` on the host: output `i` XORs
    `eq_r_dprime[b]` over the bits `b` set in `tensor[i]`, bit `b` being bit
    `b mod 64` of lane `b div 64`."""
    out = np.zeros_like(tensor_lanes)
    for i, (lo, hi) in enumerate(tensor_lanes):
        for b in range(_WIDTH):
            lane = int(hi) if b >= 64 else int(lo)
            if (lane >> (b % 64)) & 1:
                out[i] ^= eq_lanes[b]
    return out


class RsEqIndCpuTest(parameterized.TestCase):
    @parameterized.parameters(1, 2, 17, 256, 4096)
    def test_matches_the_zorch_kernel(self, n: int):
        rng = np.random.default_rng(20260909 + n)
        tensor = rand_ghash(rng, n)
        eq = rand_ghash(rng, _WIDTH)

        np.testing.assert_array_equal(
            ghash.to_lanes(_rs_eq_ind_cpu(tensor, eq)),
            ghash.to_lanes(rs_eq_ind(tensor, eq)),
        )

    def test_matches_flocks_bit_loop(self):
        """Independent of the kernel: pins the byte order the table indexes."""
        rng = np.random.default_rng(20260909)
        tensor = rand_ghash(rng, 64)
        eq = rand_ghash(rng, _WIDTH)

        np.testing.assert_array_equal(
            ghash.to_lanes(_rs_eq_ind_cpu(tensor, eq)),
            _fold_naive(ghash.to_lanes(tensor), ghash.to_lanes(eq)),
        )

    def test_zero_tensor_folds_to_zero(self):
        """No bits set selects no table entry — the fold's identity, and the
        case a padded claim's tail is entirely made of."""
        eq = rand_ghash(np.random.default_rng(1), _WIDTH)
        zeros = ghash.zeros(8)

        np.testing.assert_array_equal(
            ghash.to_lanes(_rs_eq_ind_cpu(zeros, eq)),
            np.zeros((8, 2), np.uint64),
        )

    def test_is_linear_in_the_eq_vector(self):
        """The fold is GF(2)-linear in `eq_r_dprime`, which is what lets
        `prove_batched` bake γ into the eq rather than scaling the output."""
        rng = np.random.default_rng(4)
        tensor = rand_ghash(rng, 32)
        a, b = rand_ghash(rng, _WIDTH), rand_ghash(rng, _WIDTH)

        np.testing.assert_array_equal(
            ghash.to_lanes(_rs_eq_ind_cpu(tensor, a + b)),
            ghash.to_lanes(_rs_eq_ind_cpu(tensor, a) + _rs_eq_ind_cpu(tensor, b)),
        )


if __name__ == "__main__":
    absltest.main()
