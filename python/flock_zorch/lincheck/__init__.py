"""Lincheck sub-protocol (prover) + hash-circuit seams.

Authored in `prover`, re-exported so `from flock_zorch import lincheck` and
`from flock_zorch.lincheck import CscCircuit, LincheckCircuit` resolve
unchanged. The product-sumcheck round lives in `sumcheck.inf_product`.
"""
from flock_zorch.lincheck.prover import *  # noqa: F401,F403
