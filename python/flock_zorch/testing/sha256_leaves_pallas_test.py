"""Byte-equality of the Pallas batched SHA-256 leaf hash against hash_frx.

Run (GPU required — the kernel is Triton):
  FRX_PLATFORMS=cuda,cpu PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/sha256_leaves_pallas_test.py
"""

from __future__ import annotations

import hashlib

import frx

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest  # noqa: E402

from flock_zorch.hash import merkle as fz_merkle  # noqa: E402
from flock_zorch.hash import merkle_pallas  # noqa: E402


class Sha256LeavesPallasTest(absltest.TestCase):
    def test_matches_hash_frx_and_hashlib(self) -> None:
        rng = np.random.default_rng(3)
        for b, length in ((128, 1024), (64, 64), (256, 192)):
            rows = rng.integers(0, 256, size=(b, length), dtype=np.uint8)
            got = np.asarray(merkle_pallas.sha256_leaves_pallas(fnp.asarray(rows)))
            want = np.asarray(fz_merkle._digest(fnp.asarray(rows), length))
            np.testing.assert_array_equal(got, want, err_msg=f"L={length}")
            # And against the stdlib, so a shared hash_frx bug can't hide.
            self.assertEqual(
                got[0].tobytes(), hashlib.sha256(rows[0].tobytes()).digest()
            )


if __name__ == "__main__":
    absltest.main()
