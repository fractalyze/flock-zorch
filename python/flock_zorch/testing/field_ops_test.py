"""GhashFieldOps byte-parity gate — pins flock's FieldOps seam to its bare
primitives (field.add / field.mul / sumcheck._xor_reduce), the same
NativeFieldOps-parity discipline zorch uses, so the seam and the hand-written
arithmetic cannot drift. Runs on GPU under both software and clmad mul.

Run:
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH=python:/home/jooman/fractalyze/zorch <venv> \
      python/flock_zorch/testing/field_ops_test.py
"""
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from flock_zorch import field, sumcheck  # noqa: E402
from flock_zorch.field_ops import GhashFieldOps  # noqa: E402

try:
    from flock_zorch import field_clmad
    MULS = [("software", field.mul)] + ([("clmad", field_clmad.mul)] if field_clmad.available() else [])
except Exception:  # noqa: BLE001
    MULS = [("software", field.mul)]


def _rng_f128(rng, shape):
    return rng.integers(0, 1 << 63, size=(*shape, 2), dtype=np.uint64)


def run(name, mul):
    rng = np.random.default_rng(0xF0CC)
    ops = GhashFieldOps(mul)
    a = jax.numpy.asarray(_rng_f128(rng, (37,)))
    b = jax.numpy.asarray(_rng_f128(rng, (37,)))
    tab = jax.numpy.asarray(_rng_f128(rng, (8, 5)))
    results = []

    def eq(label, got, want):
        results.append((label, np.array_equal(np.asarray(got, np.uint64), np.asarray(want, np.uint64))))

    eq("add == field.add (XOR)", ops.add(a, b), field.add(a, b))
    eq("sub == add (char 2)", ops.sub(a, b), field.add(a, b))
    eq("mul == mul", ops.mul(a, b), mul(a, b))
    eq("sum(axis=0) == _xor_reduce", ops.sum(tab, axis=0), sumcheck._xor_reduce(tab, axis=0))
    eq("sum(axis=1) == _xor_reduce", ops.sum(tab, axis=1), sumcheck._xor_reduce(tab, axis=1))
    eq("zero == [0,0]", ops.zero, np.array([0, 0], np.uint64))
    eq("one == F128::ONE [1,0]", ops.one, sumcheck.ONE)
    eq("zeros_like", ops.zeros_like(a), np.zeros_like(np.asarray(a)))

    # one is the multiplicative identity; zero is the additive identity
    eq("one is mul identity", ops.mul(ops.one, a[0]), a[0])
    eq("zero is add identity", ops.add(ops.zero, a[0]), a[0])

    raised = False
    try:
        ops.domain_point(1, a[0])
    except NotImplementedError:
        raised = True
    results.append(("domain_point raises (∞-trick owns the message)", raised))
    return results


def main() -> int:
    allok = True
    for name, mul in MULS:
        print(f"-- {name} --")
        for label, ok in run(name, mul):
            print(f"  {'PASS' if ok else 'FAIL'}  {label}")
            allok = allok and ok
    print(f"GhashFieldOps byte-parity vs flock primitives: {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
