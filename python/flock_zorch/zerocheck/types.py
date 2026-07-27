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

    What the R1CS zerocheck leaves for the lincheck: the Hadamard constraint has
    come down to evaluation claims on the three witness images, which the
    lincheck then ties back to ẑ through A and B.

    Keyword-only: `a_eval` / `b_eval` / `c_eval` are three interchangeable-looking
    values, and `mlv_challenges` / `r_rest` two interchangeable-looking coordinate
    lists — a positional swap in either group would type-check and silently open
    the wrong claim.
    """

    z: Any
    mlv_challenges: Any
    r_rest: Any
    a_eval: Any
    b_eval: Any
    c_eval: Any
