"""Zerocheck sub-protocol (prover side).

Authored in `prover`, re-exported so `from flock_zorch import zerocheck` and
`from flock_zorch.zerocheck import _lagrange_weights` resolve unchanged.
"""

from flock_zorch.types import ZerocheckClaim as ZerocheckClaim
from flock_zorch.zerocheck.prover import *  # noqa: F401,F403
from flock_zorch.zerocheck.prover import (
    # Redundant alias: marks the underscore name as a deliberate re-export
    # rather than an unused import.
    _lagrange_weights as _lagrange_weights,
)
