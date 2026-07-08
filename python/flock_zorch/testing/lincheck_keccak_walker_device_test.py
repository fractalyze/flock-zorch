"""Differential gate for the device keccak lincheck walker: the production device
fold (`_fold_walker_dev`, used by `KeccakLincheckCircuit` / `Keccak3LincheckCircuit`)
must be byte-identical to the host `accumulate_subkeccak` walker on random `eq` /
`alpha`.

Pure host (no golden, no GPU), random inputs: complements the two fixed flock
walker-probe samples in `keccak_oracle_test` / `keccak3_ligerito_oracle_test`
(stage W) with broad-coverage differential testing. `fold_alpha_batched`'s cost is
data-independent, so random eq exercises the full φᵀ/χ transpose + recurrence.
"""
from __future__ import annotations

import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)  # ghash ops at flock_zorch import need x64

from flock_zorch import keccak_lincheck as kk          # noqa: E402
from flock_zorch.keccak_lincheck import KeccakLincheckCircuit, _fold_walker_numpy  # noqa: E402
from flock_zorch import keccak3_lincheck as kk3         # noqa: E402
from flock_zorch.keccak3_lincheck import Keccak3LincheckCircuit  # noqa: E402


def _check(name, circ, sub_cols, z_const, n_cols, results, seed):
    rng = np.random.default_rng(seed)
    ok = True
    for _ in range(4):
        eq = rng.integers(0, 2**64, size=(n_cols, 2), dtype=np.uint64)
        alpha = rng.integers(0, 2**64, size=2, dtype=np.uint64)
        got = np.asarray(circ.fold_alpha_batched(alpha, eq))            # device
        want = _fold_walker_numpy(eq, alpha, sub_cols, z_const, n_cols)  # host reference
        ok = ok and got.shape == want.shape and np.array_equal(got, want)
    results.append((f"{name} device == host walker", ok))


def main() -> int:
    results = []
    _check("single-keccak", KeccakLincheckCircuit(), KeccakLincheckCircuit._sub_cols,
           kk.Z_CONST, kk.K, results, seed=1)
    _check("keccak3", Keccak3LincheckCircuit(), Keccak3LincheckCircuit._sub_cols,
           kk3.Z_CONST, kk3.K, results, seed=2)

    ok = True
    for nm, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {nm}")
        ok = ok and passed
    print(f"keccak walker device vs host byte-identity: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
