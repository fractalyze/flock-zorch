"""flock's fused R1CS prover (`prover::prove` / `prove_fast_core`), authored in
jax — byte-identical to flock-core. Chains the byte-identical phases on ONE
shared SHA-256 challenger with device-resident state (no per-phase host
re-transfer): commit → bind_statement → zerocheck → lincheck → batched PCS open.

This is the honest single-call e2e measurement (vs the standalone-phase sum in
e2e_gpu_bench) and removes the witness transfer (a=A·z, b=B·z are device-
resident). Gated by `testing/e2e_oracle_test.py` against flock `prover::prove`.
"""
from __future__ import annotations

import numpy as np

from flock_zorch.challenger import Challenger  # noqa: F401  (re-exported for callers)


def _as_bytes(x) -> bytes:
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    return np.asarray(x, np.uint8).tobytes()


def bind_statement(ch, statement_digest, root) -> None:
    """Bind the Fiat-Shamir transcript to the statement (flock `proof::bind_statement`):
    observe `flock-r1cs-v0` + the R1CS instance digest + the commitment root. Call
    once after commit, before any sub-protocol challenge."""
    ch.observe_label(b"flock-r1cs-v0")
    ch.observe_bytes(_as_bytes(statement_digest))
    ch.observe_bytes(_as_bytes(root))
