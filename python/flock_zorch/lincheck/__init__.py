"""Lincheck sub-protocol (prover) — the circuit-independent half.

Authored in `prover`, re-exported so `from flock_zorch import lincheck` and
`from flock_zorch.lincheck import CscCircuit, LincheckCircuit` resolve
unchanged. The product-sumcheck round lives in `sumcheck.inf_product`.

`LincheckCircuit` is the seam a circuit plugs into; per-circuit implementations
of it live with their circuits in `flock_zorch.r1cs_hashes`. `CscCircuit`'s
docstring says which circuits need one and which do not.
"""

from flock_zorch.lincheck.prover import *  # noqa: F401,F403
