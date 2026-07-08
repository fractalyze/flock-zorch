"""F₂¹²⁸ field arithmetic + GF(2⁸) helpers.

Public API is authored in `f128` and re-exported here so
`from flock_zorch import field` / `field.<name>` call sites resolve unchanged.
"""
from flock_zorch.field.f128 import *  # noqa: F401,F403
from flock_zorch.field.f128 import _to_int, _to_lohi  # noqa: F401  (used cross-module by lincheck)
