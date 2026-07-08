"""Zerocheck sub-protocol (prover side).

Authored in `prover`, re-exported so `from flock_zorch import zerocheck` and
`from flock_zorch.zerocheck import _lagrange_weights` resolve unchanged.
"""
from flock_zorch.zerocheck.prover import *  # noqa: F401,F403
from flock_zorch.zerocheck.prover import _lagrange_weights  # noqa: F401  (used by lincheck)
