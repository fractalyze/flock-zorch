"""The zerocheck's claim — the shared leaf both roles produce.

Kept out of `prover.py` and `verifier.py` so neither has to import the other:
the claim is what the reduction *means*, and both roles state the same one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class ZerocheckClaim:
    """â, b̂ and ĉ evaluate to `a_eval`, `b_eval`, `c_eval` at the point the
    zerocheck drew — the skip scalar `z` together with the coordinate lists.

    Keyword-only: the three evals, and the two coordinate lists, are each
    interchangeable-looking, so a positional swap would type-check.
    """

    z: Any
    mlv_challenges: Any
    r_rest: Any
    a_eval: Any
    b_eval: Any
    c_eval: Any
