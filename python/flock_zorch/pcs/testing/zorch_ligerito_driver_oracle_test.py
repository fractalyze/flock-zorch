"""zorch-driver Ligerito byte gate vs flock (flock-zorch#32 T4, driver level).

Drives `zorch.pcs.ligerito` — code-generic recursion, flock wire via the
`FlockChoreography`/`FlockTranscript` seams, flock algebra via
`monomial_commit` + the raw-basis entry, witness/basis bit-reversed at entry —
over the `dump_ligerito` golden, and byte-compares against flock-core:

  - every transcript-visible proof field vs the golden: initial + recursive
    roots, all sumcheck messages, OOD values, PoW nonces, the residual, and
    each level's opened rows;
  - the verifier accepts and its Fiat-Shamir byte stream mirrors the prover's.

The Merkle multi-proof bytes are flock's octopus assembly — a proof-container
format, not a transcript input — so this driver-level gate does not cover them;
`ligerito_oracle_test` byte-checks the whole flock `LigeritoProof` (octopus
included) produced by `zorch_ligerito.prove_flock_ligerito` over the same golden.

Run:
  FRX_PLATFORMS=cpu PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" \
      .venv/bin/python python/flock_zorch/pcs/testing/zorch_ligerito_driver_oracle_test.py
"""
import sys
from pathlib import Path

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
from frx import lax  # noqa: E402

from zorch.coding.reed_solomon import ReedSolomon  # noqa: E402
from zorch.pcs.ligerito.prover import LigeritoProver  # noqa: E402
from zorch.pcs.ligerito.verifier import LigeritoVerifier  # noqa: E402

from flock_zorch.hash import merkle  # noqa: E402
from flock_zorch.pcs.ligerito import (  # noqa: E402
    flock_ligerito_config,
    flock_transcript,
)

ART = Path(__file__).resolve().parents[4] / "artifacts"
DOMAIN = b"flock-ligerito-test"


class R:
    def __init__(self, buf): self.b = buf; self.o = 0
    def take(self, n): v = self.b[self.o:self.o + n]; self.o += n; return v
    def u(self): return int.from_bytes(self.take(8), "little")
    def uv(self): return [self.u() for _ in range(self.u())]
    def f(self): return np.frombuffer(self.take(16), np.uint64).copy()
    def fv(self): n = self.u(); return np.frombuffer(self.take(16 * n), np.uint64).reshape(n, 2).copy()
    def raw(self, n): return np.frombuffer(self.take(n), np.uint8).copy()
    def hv(self): n = self.u(); return np.frombuffer(self.take(32 * n), np.uint8).reshape(n, 32).copy()
    def u64v(self): return [self.u() for _ in range(self.u())]
    def rows(self): n = self.u(); return [self.fv() for _ in range(n)]
    def pair(self): n = self.u(); return [(self.f(), self.f()) for _ in range(n)]


def load():
    rd = R((ART / "ligerito_golden.bin").read_bytes())
    assert bytes(rd.take(8)) == b"FLKLIG01", "bad magic"
    g = dict(log_n=rd.u(), m=rd.u(), lbs=rd.u())
    g["cfg"] = dict(
        log_inv_rates=rd.uv(), recursive_steps=rd.u(), initial_log_msg_cols=rd.u(),
        initial_log_num_interleaved=rd.u(), initial_k=rd.u(), recursive_log_msg_cols=rd.uv(),
        recursive_ks=rd.uv(), queries=rd.uv(), grinding_bits=rd.uv(),
        fold_grinding_bits=rd.uv(), ood_samples=rd.uv())
    g["f"] = rd.fv(); g["b"] = rd.fv(); g["target"] = rd.f()
    g["l0_codeword"] = rd.fv(); g["l0_tree"] = rd.hv()
    g["initial_root"] = rd.raw(32)
    g["initial_proof"] = dict(opened_rows=rd.rows(), merkle_proof=rd.hv())
    g["recursive_roots"] = rd.hv()
    nrp = rd.u(); g["recursive_proofs"] = [dict(opened_rows=rd.rows(), merkle_proof=rd.hv()) for _ in range(nrp)]
    g["final_proof"] = dict(yr=rd.fv(), opened_rows=rd.rows(), merkle_proof=rd.hv())
    g["sumcheck_transcript"] = rd.pair()
    g["grinding_nonces"] = rd.u64v()
    g["ood_values"] = rd.fv()
    g["fold_grinding_nonces"] = rd.u64v()
    return g


def _ghash(lohi) -> fnp.ndarray:
    return lax.bitcast_convert_type(
        fnp.asarray(np.asarray(lohi, np.uint64)), fnp.binary_field_ghash
    )


def _lohi(x) -> np.ndarray:
    b = np.asarray(lax.bitcast_convert_type(x, fnp.uint8))
    return np.frombuffer(b.tobytes(), np.uint64).reshape(-1, 2)


def _bitrev(x: fnp.ndarray) -> fnp.ndarray:
    return lax.bit_reverse(x, dimensions=(0,))


def _first_divergence(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


results = []


def check(name: str, ok: bool, detail: str = ""):
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else ""))


def main() -> int:
    g = load()
    cfg = g["cfg"]
    config, chor = flock_ligerito_config(cfg, g["log_n"])
    print(f"device {frx.devices()[0]} | log_n={g['log_n']} fold_ks={config.fold_ks} "
          f"queries={config.queries} ood={config.ood_samples}")

    def make_code(message_len: int, log_inv_rate: int) -> ReedSolomon:
        return ReedSolomon(message_len=message_len, blowup=1 << log_inv_rate,
                           dtype=fnp.binary_field_ghash)

    prover = LigeritoProver(make_code, merkle.GHASH_TREE, config, chor)
    verifier = LigeritoVerifier(make_code, merkle.GHASH_TREE, config, chor)

    W = _bitrev(_ghash(g["f"]))
    B = _bitrev(_ghash(g["b"]))
    target = _ghash(g["target"][None])[0]

    root, pdata = prover.commit([W])
    check("initial_root", np.array_equal(np.asarray(root), g["initial_root"]))

    proof, t_open = prover.open_with_basis(pdata, B, target, flock_transcript(DOMAIN))

    # --- transcript-visible proof fields vs the golden ---
    golden_msgs = np.stack([np.stack(p) for p in g["sumcheck_transcript"]])
    zorch_msgs = np.stack([_lohi(m) for m in proof.sumcheck_messages])
    check("sumcheck messages", np.array_equal(zorch_msgs, golden_msgs))
    check(
        "recursive roots",
        np.array_equal(
            np.stack([np.asarray(r) for r in proof.recursive_roots]),
            g["recursive_roots"],
        ),
    )
    check(
        "ood values",
        np.array_equal(
            np.concatenate([_lohi(y) for y in proof.ood_values]), g["ood_values"]
        ),
    )
    check("residual (yr)", np.array_equal(_lohi(_bitrev(proof.final_residual)),
                                          g["final_proof"]["yr"]))

    # PoW witnesses ride in schedule order; the golden splits them into
    # fold_grinding_nonces + grinding_nonces — re-interleave per the schedule.
    fold_it = iter(g["fold_grinding_nonces"])
    query_it = iter(g["grinding_nonces"])
    expected_pow = []
    for j in range(config.num_levels):
        for i in range(config.fold_ks[j]):
            if chor.fold_grind_bits(j, i) is not None:
                expected_pow.append(next(fold_it))
        if chor.query_grind_bits(j) is not None:
            expected_pow.append(next(query_it))
    check("pow witnesses", [int(w) for w in proof.pow_witnesses] == expected_pow)

    opened_golden = [g["initial_proof"]["opened_rows"]] + [
        rp["opened_rows"] for rp in g["recursive_proofs"]
    ] + [g["final_proof"]["opened_rows"]]
    rows_ok = True
    for j, opening in enumerate(proof.component_openings):
        got = _lohi(opening.row).reshape(opening.row.shape[0], -1, 2)
        want = np.stack(opened_golden[j])
        rows_ok = rows_ok and np.array_equal(got, want)
    check("opened rows (all levels)", rows_ok)

    # --- verifier: accepts, and its FS stream mirrors the prover's ---
    ok, t_verify = verifier.verify_with_basis(root, B, target, proof,
                                              flock_transcript(DOMAIN))
    check("verify ok", bool(ok))
    div = _first_divergence(t_verify.inner.buffer, t_open.inner.buffer)
    check("verifier FS stream", div == -1, f"first divergence at byte {div}")

    allok = all(ok for _, ok in results)
    print(f"zorch-driver ligerito vs flock golden: {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
