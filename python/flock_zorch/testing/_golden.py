# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared golden-fixture IO and proof comparison for the byte gates.

The gates all ingest the same little-endian wire the `dump_*` Rust examples
emit, and the Ligerito ones all compare the same `R1csProofLigerito` field set.
Both were copy-pasted per gate: the reader nine times, the proof comparison
seven, each with its own hardcoded `parents[N]` walk up to `artifacts/`. This is
the one copy.

Nothing here knows about a specific circuit — per-circuit facts (the magic
bytes, which optional sections a dump carries) stay with the gate that owns
them.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import flock_zorch


def artifacts_dir() -> Path:
    """The `artifacts/` directory holding the goldens.

    Anchored on the installed `flock_zorch` package rather than a per-file
    `parents[N]` walk: N differed by nesting depth (3 for `testing/`, 4 for
    `<pkg>/testing/`) and was hardcoded in 17 places, so moving a gate one
    directory made it silently read the wrong tree. `FLOCK_ZORCH_ARTIFACTS`
    overrides, which previously only 4 of those 17 honored.
    """
    if env := os.environ.get("FLOCK_ZORCH_ARTIFACTS"):
        return Path(env)
    # flock_zorch/__init__.py -> flock_zorch/ -> python/ -> repo root
    return Path(flock_zorch.__file__).resolve().parents[2] / "artifacts"


ART = artifacts_dir()


class R:
    """Cursor over a `dump_*` golden. Every field is little-endian; `f`/`fv` read
    F128 as the uint64 [lo, hi] lane pair that is also its wire form."""

    def __init__(self, buf): self.b = buf; self.o = 0
    def take(self, n): v = self.b[self.o:self.o + n]; self.o += n; return v
    def u(self): return int.from_bytes(self.take(8), "little")
    def uv(self): return [self.u() for _ in range(self.u())]
    def u64v(self): return [self.u() for _ in range(self.u())]
    def f(self): return np.frombuffer(self.take(16), np.uint64).copy()
    def fv(self): n = self.u(); return np.frombuffer(self.take(16 * n), np.uint64).reshape(n, 2).copy()
    def pair(self): n = self.u(); return [(self.f(), self.f()) for _ in range(n)]
    def raw(self, n): return np.frombuffer(self.take(n), np.uint8).copy()
    def hv(self): n = self.u(); return np.frombuffer(self.take(32 * n), np.uint8).reshape(n, 32).copy()
    def rowsf(self): n = self.u(); return [self.fv() for _ in range(n)]
    def rowsu(self): n = self.u(); return [np.frombuffer(self.take(4 * self.u()), np.uint32).copy() for _ in range(n)]

    # `pcs/testing` spelled the F128-rows reader `rows`; keep both names so those
    # gates read unchanged.
    rows = rowsf


def open_golden(name: str) -> R:
    """Reader over `artifacts/<name>`, with a message naming the dump example
    when it is absent — `artifacts/` is gitignored and the goldens are produced
    on demand, so missing is the normal state on a fresh checkout."""
    path = ART / name
    try:
        return R(path.read_bytes())
    except FileNotFoundError:
        # Don't guess the example name from the filename — the m-variant dumps
        # (`..._golden_m24.bin`) don't follow it, and a wrong command is worse
        # than none. `scripts/dump_goldens.sh` and `examples/dump_*.rs` are the
        # index.
        raise SystemExit(
            f"missing golden {path}\n"
            f"  the goldens are gitignored and dumped on demand — regenerate with the\n"
            f"  matching `cargo run --release --example dump_*` (see examples/) or\n"
            f"  `scripts/dump_goldens.sh`") from None


def unpack_bits(packed, m: int):
    """Packed F128 witness (uint64 [n, 2]) -> the uint8 0/1 bit array of length
    2^m. The host twin of `prover._unpack_bits`, which is the jitted device form
    and does not trim to 2^m."""
    packed = np.asarray(packed, np.uint64).reshape(-1, 2)
    bi = np.arange(64, dtype=np.uint64)
    lo = ((packed[:, 0:1] >> bi) & np.uint64(1)).astype(np.uint8)
    hi = ((packed[:, 1:2] >> bi) & np.uint64(1)).astype(np.uint8)
    return np.concatenate([lo, hi], axis=1).reshape(-1)[: 1 << m]


# --------------------------------------------------------------- wire sections

def read_ligerito_config(rd: R) -> dict:
    """The Ligerito prover config every `dump_*_ligerito` golden carries."""
    return dict(log_inv_rates=rd.uv(), recursive_steps=rd.u(), initial_log_msg_cols=rd.u(),
                initial_log_num_interleaved=rd.u(), initial_k=rd.u(),
                recursive_log_msg_cols=rd.uv(), recursive_ks=rd.uv(), queries=rd.uv(),
                grinding_bits=rd.uv(), fold_grinding_bits=rd.uv(), ood_samples=rd.uv())


def read_ligerito_proof(rd: R) -> dict:
    """The serialized `LigeritoProof` trailer: initial commitment + opening, the
    recursive rounds, the final round, and the sumcheck/grinding transcript."""
    lig = dict(initial_root=rd.raw(32))
    lig["initial_proof"] = dict(opened_rows=rd.rowsf(), merkle_proof=rd.hv())
    lig["recursive_roots"] = rd.hv()
    nrp = rd.u()
    lig["recursive_proofs"] = [dict(opened_rows=rd.rowsf(), merkle_proof=rd.hv())
                               for _ in range(nrp)]
    lig["final_proof"] = dict(yr=rd.fv(), opened_rows=rd.rowsf(), merkle_proof=rd.hv())
    lig["sumcheck_transcript"] = rd.pair()
    lig["grinding_nonces"] = rd.u64v()
    lig["ood_values"] = rd.fv()
    lig["fold_grinding_nonces"] = rd.u64v()
    return lig


# ------------------------------------------------------------------ comparison

def _pairs(t):
    return (np.array([np.concatenate([a, b]) for a, b in t]) if t
            else np.zeros((0, 4), np.uint64))


def _rows_eq(a, b):
    return len(a) == len(b) and all(
        np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def _stk(v):
    return (np.stack([np.asarray(x).reshape(2) for x in v]) if len(v)
            else np.zeros((0, 2), np.uint64))


def ligerito_proof_results(p, gl, prefix: str = "lig ") -> list[tuple[str, bool]]:
    """`(name, matched)` for every field of a `LigeritoProof` against its golden.

    `p` is the produced proof dict, `gl` the golden's. Every gate compares the
    same field set in the same order, so a new proof field is added here once and
    every gate starts checking it.
    """
    return [
        (f"{prefix}initial_root",
         np.array_equal(p["initial_root"], gl["initial_root"])),
        (f"{prefix}sumcheck_transcript",
         np.array_equal(_pairs(p["sumcheck_transcript"]), _pairs(gl["sumcheck_transcript"]))),
        (f"{prefix}recursive_roots",
         np.array_equal(np.asarray(p["recursive_roots"]), gl["recursive_roots"])),
        (f"{prefix}ood_values", np.array_equal(_stk(p["ood_values"]), gl["ood_values"])),
        (f"{prefix}grinding_nonces",
         list(map(int, p["grinding_nonces"])) == list(gl["grinding_nonces"])),
        (f"{prefix}fold_grinding_nonces",
         list(map(int, p["fold_grinding_nonces"])) == list(gl["fold_grinding_nonces"])),
        (f"{prefix}initial_proof.opened_rows",
         _rows_eq(p["initial_proof"]["opened_rows"], gl["initial_proof"]["opened_rows"])),
        (f"{prefix}initial_proof.merkle_proof",
         np.array_equal(p["initial_proof"]["merkle_proof"], gl["initial_proof"]["merkle_proof"])),
        (f"{prefix}recursive_proofs", _recursive_proofs_eq(p, gl)),
        (f"{prefix}final_proof.yr",
         np.array_equal(np.asarray(p["final_proof"]["yr"]), gl["final_proof"]["yr"])),
        (f"{prefix}final_proof.opened_rows",
         _rows_eq(p["final_proof"]["opened_rows"], gl["final_proof"]["opened_rows"])),
        (f"{prefix}final_proof.merkle_proof",
         np.array_equal(p["final_proof"]["merkle_proof"], gl["final_proof"]["merkle_proof"])),
    ]


def _recursive_proofs_eq(p, gl) -> bool:
    if len(p["recursive_proofs"]) != len(gl["recursive_proofs"]):
        return False
    return all(_rows_eq(pr["opened_rows"], gr["opened_rows"])
               and np.array_equal(pr["merkle_proof"], gr["merkle_proof"])
               for pr, gr in zip(p["recursive_proofs"], gl["recursive_proofs"]))
