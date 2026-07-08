"""Unit test for the `lincheck.LincheckCircuit` Protocol: every circuit that
`lincheck.prove(circuit=)` accepts must match the seam structurally.

Pure host (no golden, no GPU): imports the three circuits, asserts each is a
`LincheckCircuit` (member presence, via `@runtime_checkable`), and asserts a
circuit missing either member is rejected — so a future circuit that forgets
`fold_alpha_batched` or `const_pin` fails here instead of at a `prove` call site.
The fold's math is pinned by the byte-match oracle gates, not here.
"""
from __future__ import annotations

import sys

import jax

jax.config.update("jax_enable_x64", True)  # ghash ops at flock_zorch import need x64

from flock_zorch.lincheck import CscCircuit, LincheckCircuit  # noqa: E402
from flock_zorch.lincheck.keccak import KeccakLincheckCircuit  # noqa: E402
from flock_zorch.lincheck.keccak3 import Keccak3LincheckCircuit  # noqa: E402


class _MissingFold:
    const_pin = None


class _MissingConstPin:
    def fold_alpha_batched(self, alpha, eq_inner):
        return None


def _check(name, obj, want, results):
    got = isinstance(obj, LincheckCircuit)
    results.append((f"{name} isinstance LincheckCircuit == {want}", got == want))


def main() -> int:
    results = []
    # The three real circuits conform (CscCircuit takes empty sparse rows here —
    # construction only, no fold is run).
    _check("CscCircuit", CscCircuit([], [], 8, const_pin=None), True, results)
    _check("KeccakLincheckCircuit", KeccakLincheckCircuit(), True, results)
    _check("Keccak3LincheckCircuit", Keccak3LincheckCircuit(), True, results)
    # Negative cases: a missing member must be rejected.
    _check("_MissingFold", _MissingFold(), False, results)
    _check("_MissingConstPin", _MissingConstPin(), False, results)

    ok = True
    for nm, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {nm}")
        ok = ok and passed
    print(f"LincheckCircuit Protocol conformance: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
