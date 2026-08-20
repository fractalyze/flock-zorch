# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Python-native gate (no golden): `s_hat_v_from_z_vec` equals the
`bit_slice_evals` witness read for the ab claim.

The two derivations of the ab claim's s_hat_v — bit-slicing the full packed
witness against eq(x_full[1:]) vs eq(tail)-folding the lincheck's initial
z_vec — are the same GF(2¹²⁸) sum reassociated, so they must match
bit-for-bit. Mirrors flock's `s_hat_v_from_z_vec_matches_fold_1b_rows_ab`:
witness → `pack_witness` / `pack_z_lincheck_from_packed` → the real
`partial_fold_packed_z` producer, compared per (m, k_log) case including the
degenerate k_log == LOG_PACKING (empty tail: z_vec IS s_hat_v)."""

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zorch.pcs.ring_switch import bit_slice_evals

from flock_zorch import ghash, sumcheck
from flock_zorch.lincheck import partial_fold_packed_z
from flock_zorch.pcs.pack import pack_witness, pack_z_lincheck_from_packed
from flock_zorch.pcs.ring_switch import LOG_PACKING, s_hat_v_from_z_vec
from flock_zorch.testing._util import rand_ghash

# K_SKIP is fixed at 6 (so tail = r_inner_rest[1:] has k_log − 7 coords;
# r_inner_rest[0] is ring-switch's x_full[1:] prefix bit because
# K_SKIP + 1 = LOG_PACKING). n_log = m − k_log ≥ 3 for the stripe layout.
_K_SKIP = 6
_CASES = [(13, 10), (15, 11), (17, 13), (13, LOG_PACKING)]  # last: empty tail


class SHatVFromZVecTest(absltest.TestCase):
    def test_matches_bit_slice_evals(self):
        rng = np.random.default_rng(20260820)
        for m, k_log in _CASES:
            with self.subTest(m=m, k_log=k_log):
                z_bits = rng.integers(0, 2, size=1 << m, dtype=np.uint8)
                z_packed = pack_witness(z_bits, m)
                z_lincheck = pack_z_lincheck_from_packed(z_packed, m, k_log)

                r_inner_rest = rand_ghash(rng, k_log - _K_SKIP)
                x_outer = rand_ghash(rng, m - k_log)

                # The witness-read derivation: eq over the full ab suffix
                # x_full[1:] = r_inner_rest[1:] ++ x_outer.
                suffix = fnp.concatenate([r_inner_rest[1:], x_outer], axis=0)
                want = bit_slice_evals(
                    ghash.to_ghash(fnp.asarray(z_packed)), sumcheck.build_eq(suffix)
                )

                # The z_vec derivation: the real lincheck producer, then the
                # eq(tail) fold.
                z_vec = partial_fold_packed_z(z_lincheck, m, k_log, x_outer)
                got = s_hat_v_from_z_vec(z_vec, r_inner_rest[1:])

                np.testing.assert_array_equal(ghash.to_lanes(got), ghash.to_lanes(want))


if __name__ == "__main__":
    absltest.main()
