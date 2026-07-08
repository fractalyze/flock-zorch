"""Lincheck sub-protocol (prover) + hash-circuit seams.

Authored in `prover`, re-exported so `from flock_zorch import lincheck`,
`from flock_zorch.lincheck import CscCircuit, LincheckCircuit`, and
`from flock_zorch.lincheck import _round_eval, _bind_top` resolve unchanged.
"""
from flock_zorch.lincheck.prover import *  # noqa: F401,F403
from flock_zorch.lincheck.prover import _round_eval, _bind_top  # noqa: F401  (used by chain)
