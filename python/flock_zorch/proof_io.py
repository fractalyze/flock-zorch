# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""flock's proof-bundle wire format (`proof_io::R1csProofBundleLigerito`).

The benchmark harness consumes a proof as one file: `b"FLOCK" ‖ version=4 ‖
flavor=2`, then bincode 1.x with fixint encoding, little-endian — `usize` as
u64, a `Vec` as a u64 count followed by its elements, an enum as its u32
variant index, `[u8; 32]` raw, and `F128` as `lo_le8 ‖ hi_le8` (identical
bytes to the ghash lanes). The structs, in file order:

    Commitment { root: [u8;32], params: PcsParams { m, log_inv_rate,
                 log_batch_size: u64 ×3, profile: enum, merkle_hash: enum } }
    ZerocheckProof { round1_ab: Vec<F128>, round1_c: Vec<F128>,
                     multilinear_rounds: Vec<(F128, F128)>,
                     final_a_eval, final_b_eval, final_c_eval: F128 ×3 }
    LincheckProof { rounds: Vec<(F128, F128)>, z_partial: Vec<F128> }
    BatchOpeningProofLigerito {
      ring_switches: Vec<RingSwitchProof { s_hat_v: Vec<F128> }>,
      ligerito: LigeritoProof {
        initial_root: [u8;32], initial_proof: RecursiveProof,
        recursive_roots: Vec<[u8;32]>, recursive_proofs: Vec<RecursiveProof>,
        final_proof: FinalProof { yr: Vec<F128>, opened_rows, merkle_proof },
        sumcheck_transcript: Vec<SumcheckMessage { u_0, u_2 }>,
        grinding_nonces: Vec<u64>, ood_values: Vec<F128>,
        fold_grinding_nonces: Vec<u64> } }
    RecursiveProof { opened_rows: Vec<Vec<F128>>, merkle_proof: Vec<[u8;32]> }

CAUTION — two `HashKind` enums with OPPOSITE orders exist in the fork:
`PcsParams.merkle_hash` (serialized here) is Sha256=0/Blake3=1, while the
chain-bundle flavor discriminator in `proof_io` is Blake3=0/Sha2=1/Keccak=2.
Swapping them deserializes cleanly and fails only at verify.

The writer takes the prover's own objects — ghash arrays or their uint64-lane
form both serialize identically (`ghash.tobytes()` IS the wire) — plus the
`ligerito` wire dict `pcs.ligerito.prove_flock_ligerito` assembles.
`parse_bundle` is the exact inverse, for the byte gates' field-level diffs.

Byte-gated end to end against a bundle produced and verified by the
fork's own crates (`testing/proof_io_test.py` carries the round-trip pins;
the full-prove gate compares a whole serialized bundle to the fork's).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from flock_zorch import ghash

MAGIC = b"FLOCK"
VERSION = 4
FLAVOR_LIGERITO = 2

PROFILE_FAST = 0
PROFILE_SLIM = 1
PROFILE_SECURE = 2
# PcsParams' HashKind order (NOT the chain-bundle discriminator's).
PARAMS_MERKLE_SHA256 = 0
PARAMS_MERKLE_BLAKE3 = 1


@dataclasses.dataclass(frozen=True)
class PcsParams:
    """The commitment's parameter block, as the wire carries it."""

    m: int
    log_inv_rate: int
    log_batch_size: int
    profile: int = PROFILE_FAST
    merkle_hash: int = PARAMS_MERKLE_SHA256


def _u64(n) -> bytes:
    return int(n).to_bytes(8, "little")


def _u32(n) -> bytes:
    return int(n).to_bytes(4, "little")


def _lanes(x) -> np.ndarray:
    """ghash array or uint64-lane array -> uint64 `[n, 2]` lanes."""
    a = np.asarray(x)
    if a.dtype == np.uint64:
        return np.ascontiguousarray(a).reshape(-1, 2)
    return np.ascontiguousarray(ghash.to_lanes(a)).reshape(-1, 2)


def _f128(x) -> bytes:
    lanes = _lanes(x)
    if lanes.shape[0] != 1:
        raise ValueError(f"expected one F128, got {lanes.shape[0]}")
    return lanes.tobytes()


def _vec_f128(x) -> bytes:
    lanes = _lanes(x)
    return _u64(lanes.shape[0]) + lanes.tobytes()


def _vec_pairs(rounds) -> bytes:
    """`Vec<(F128, F128)>` from a sequence of message pairs (each a 2-tuple of
    elements or one 2-element array) or one `[n, 2]` array."""
    if isinstance(rounds, (list, tuple)):
        out = [_u64(len(rounds))]
        for r in rounds:
            if isinstance(r, (list, tuple)):
                out.append(_f128(r[0]) + _f128(r[1]))
            else:
                pair = _lanes(r)
                if pair.shape[0] != 2:
                    raise ValueError(f"expected an (F128, F128) pair, got {pair.shape}")
                out.append(pair.tobytes())
        return b"".join(out)
    lanes = _lanes(rounds).reshape(-1, 2, 2)
    return _u64(lanes.shape[0]) + lanes.tobytes()


def _root(x) -> bytes:
    a = np.ascontiguousarray(np.asarray(x), dtype=np.uint8).reshape(-1)
    if a.shape != (32,):
        raise ValueError(f"expected a 32-byte root, got shape {a.shape}")
    return a.tobytes()


def _vec_roots(roots) -> bytes:
    return _u64(len(roots)) + b"".join(_root(r) for r in roots)


def _vec_u64(ns) -> bytes:
    return _u64(len(ns)) + b"".join(_u64(n) for n in ns)


def _recursive_proof(level: dict) -> bytes:
    """`RecursiveProof` from a `prove_flock_ligerito` level dict
    (`opened_rows`: per-row lane arrays; `merkle_proof`: uint8 `[n, 32]`)."""
    rows = level["opened_rows"]
    out = [_u64(len(rows))]
    out.extend(_vec_f128(row) for row in rows)
    proof = np.ascontiguousarray(np.asarray(level["merkle_proof"]), dtype=np.uint8)
    out.append(_u64(proof.shape[0]) + proof.tobytes())
    return b"".join(out)


def bundle_bytes(
    root,
    params: PcsParams,
    zerocheck,
    lincheck_rounds,
    z_partial,
    ring_switches,
    ligerito: dict,
) -> bytes:
    """Serialize one proof into the harness's bundle file bytes.

    `zerocheck` is the prover's `ZerocheckProof`; `lincheck_rounds`/`z_partial`
    the lincheck proof's two wire fields; `ring_switches` the batched open's
    `s_hat_v` list; `ligerito` the wire dict from `prove_flock_ligerito`."""
    out = [MAGIC, bytes([VERSION]), bytes([FLAVOR_LIGERITO])]
    # Commitment
    out.append(_root(root))
    out.append(_u64(params.m) + _u64(params.log_inv_rate) + _u64(params.log_batch_size))
    out.append(_u32(params.profile) + _u32(params.merkle_hash))
    # ZerocheckProof
    out.append(_vec_f128(zerocheck.round1_ab))
    out.append(_vec_f128(zerocheck.round1_c))
    out.append(_vec_pairs(zerocheck.multilinear_rounds))
    out.append(_f128(zerocheck.final_a_eval))
    out.append(_f128(zerocheck.final_b_eval))
    out.append(_f128(zerocheck.final_c_eval))
    # LincheckProof
    out.append(_vec_pairs(lincheck_rounds))
    out.append(_vec_f128(z_partial))
    # BatchOpeningProofLigerito
    out.append(_u64(len(ring_switches)))
    out.extend(_vec_f128(s) for s in ring_switches)
    out.append(_root(ligerito["initial_root"]))
    out.append(_recursive_proof(ligerito["initial_proof"]))
    out.append(_vec_roots(ligerito["recursive_roots"]))
    out.append(_u64(len(ligerito["recursive_proofs"])))
    out.extend(_recursive_proof(p) for p in ligerito["recursive_proofs"])
    fp = ligerito["final_proof"]
    out.append(_vec_f128(fp["yr"]))
    out.append(_recursive_proof(fp))
    out.append(_vec_pairs(ligerito["sumcheck_transcript"]))
    out.append(_vec_u64(ligerito["grinding_nonces"]))
    ood = ligerito["ood_values"]
    out.append(_u64(len(ood)) + b"".join(_f128(v) for v in ood))
    out.append(_vec_u64(ligerito["fold_grinding_nonces"]))
    return b"".join(out)


class _Reader:
    def __init__(self, data: bytes):
        self._b, self._o = data, 0

    def take(self, n: int) -> bytes:
        v = self._b[self._o : self._o + n]
        if len(v) != n:
            raise ValueError(f"bundle truncated at offset {self._o}")
        self._o += n
        return v

    def u64(self) -> int:
        return int.from_bytes(self.take(8), "little")

    def u32(self) -> int:
        return int.from_bytes(self.take(4), "little")

    def f128s(self, n: int) -> np.ndarray:
        return np.frombuffer(self.take(16 * n), np.uint64).reshape(n, 2)

    def vec_f128(self) -> np.ndarray:
        return self.f128s(self.u64())

    def vec_pairs(self) -> np.ndarray:
        n = self.u64()
        return np.frombuffer(self.take(32 * n), np.uint64).reshape(n, 2, 2)

    def recursive_proof(self) -> dict:
        rows = [self.vec_f128() for _ in range(self.u64())]
        proof = np.frombuffer(self.take(32 * self.u64()), np.uint8).reshape(-1, 32)
        return {"opened_rows": rows, "merkle_proof": proof}

    def done(self) -> bool:
        return self._o == len(self._b)


def parse_bundle(data: bytes) -> dict:
    """Exact inverse of `bundle_bytes` — the byte gates' field-level diff and
    the reference reading of the wire map above."""
    rd = _Reader(data)
    if (
        rd.take(5) != MAGIC
        or rd.take(1)[0] != VERSION
        or rd.take(1)[0] != FLAVOR_LIGERITO
    ):
        raise ValueError("not a version-4 Ligerito-flavor FLOCK bundle")
    out: dict[str, Any] = {"root": rd.take(32)}
    out["params"] = PcsParams(
        m=rd.u64(),
        log_inv_rate=rd.u64(),
        log_batch_size=rd.u64(),
        profile=rd.u32(),
        merkle_hash=rd.u32(),
    )
    out["zc_round1_ab"] = rd.vec_f128()
    out["zc_round1_c"] = rd.vec_f128()
    out["zc_multilinear"] = rd.vec_pairs()
    out["zc_finals"] = rd.f128s(3)
    out["lc_rounds"] = rd.vec_pairs()
    out["lc_z_partial"] = rd.vec_f128()
    out["ring_switches"] = [rd.vec_f128() for _ in range(rd.u64())]
    out["lig_initial_root"] = rd.take(32)
    out["lig_initial_proof"] = rd.recursive_proof()
    out["lig_recursive_roots"] = [rd.take(32) for _ in range(rd.u64())]
    out["lig_recursive_proofs"] = [rd.recursive_proof() for _ in range(rd.u64())]
    out["lig_final_yr"] = rd.vec_f128()
    out["lig_final_proof"] = rd.recursive_proof()
    out["lig_sumcheck"] = rd.vec_pairs()
    out["lig_grinding_nonces"] = [rd.u64() for _ in range(rd.u64())]
    out["lig_ood_values"] = rd.vec_f128()
    out["lig_fold_grinding_nonces"] = [rd.u64() for _ in range(rd.u64())]
    if not rd.done():
        raise ValueError("trailing bytes after the bundle")
    return out
