"""`_commit_prep` must equal the eager `_bitrev(to_ghash(z))` byte for byte.

Both operations are reinterpretations — a bitcast and an index permutation — so
tracing them together changes only where they run. The commit root depends on
the result, so any drift here changes the proof.
"""

import sys

import frx

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
import numpy as np  # noqa: E402

from flock_zorch import ghash  # noqa: E402
from flock_zorch.pcs import ligerito  # noqa: E402
from flock_zorch.testing._util import report  # noqa: E402


def run():
    checks = []
    for log_n in (4, 10, 14):
        rng = np.random.default_rng(log_n)
        z = fnp.asarray(rng.integers(0, 2**64, size=(1 << log_n, 2), dtype=np.uint64))
        want = ligerito._bitrev(ghash.to_ghash(z))
        got = ligerito._commit_prep(z)
        checks.append(
            (
                f"commit prep 2^{log_n}",
                np.array_equal(np.asarray(got), np.asarray(want)),
            )
        )
    return checks


if __name__ == "__main__":
    sys.exit(report(run(), "commit prep traced vs eager"))
