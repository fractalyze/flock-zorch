"""GPU Ligerito recursive PCS byte gate vs flock (driver-isolated).

Ingests the dump_ligerito golden (config + synthetic f/b/target + L0 commit + full
LigeritoProof) and byte-compares the whole flock `LigeritoProof` — every field,
including the octopus `merkle_proof` — produced by `zorch_ligerito.prove_flock_ligerito`
(the flock-zorch prove path: `zorch.pcs.ligerito` driven through the flock FS seam,
octopus reassembled from the zorch openings). This is the golden that survived
retiring the in-tree frx port (flock-zorch#32 T5); the transcript-visible fields
are also gated at the driver level by `zorch_ligerito_driver_oracle_test`.

Run (regen golden: cargo run --release --example dump_ligerito -- 15 artifacts/ligerito_golden.bin):
  JAX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/pcs/testing/ligerito_oracle_test.py
"""
import sys
from pathlib import Path

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch import ghash  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402

ART = Path(__file__).resolve().parents[4] / "artifacts"


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
    # LigeritoProof
    g["initial_proof"] = dict(opened_rows=rd.rows(), merkle_proof=rd.hv())
    g["recursive_roots"] = rd.hv()
    nrp = rd.u(); g["recursive_proofs"] = [dict(opened_rows=rd.rows(), merkle_proof=rd.hv()) for _ in range(nrp)]
    g["final_proof"] = dict(yr=rd.fv(), opened_rows=rd.rows(), merkle_proof=rd.hv())
    g["sumcheck_transcript"] = rd.pair()
    g["grinding_nonces"] = rd.u64v()
    g["ood_values"] = rd.fv()
    g["fold_grinding_nonces"] = rd.u64v()
    return rd, g


def run():
    _, g = load()
    cfg = g["cfg"]
    results = []

    # The flock-zorch prove path → byte-gate every LigeritoProof field.
    ch = Challenger(b"flock-ligerito-test")
    _root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, g["f"])
    p = zorch_ligerito.prove_flock_ligerito(cfg, pdata, ghash.to_ghash(g["b"]),
                                            ghash.to_ghash(g["target"]), ch)

    def pairs(t): return np.array([np.concatenate([a, b]) for a, b in t]) if t else np.zeros((0, 4), np.uint64)
    def rows_eq(a, b): return len(a) == len(b) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
    def stk(v): return np.stack([np.asarray(x).reshape(2) for x in v]) if len(v) else np.zeros((0, 2), np.uint64)

    results.append(("initial_root", np.array_equal(p["initial_root"], g["initial_root"])))
    results.append(("sumcheck_transcript", np.array_equal(pairs(p["sumcheck_transcript"]), pairs(g["sumcheck_transcript"]))))
    results.append(("recursive_roots", np.array_equal(np.asarray(p["recursive_roots"]), g["recursive_roots"])))
    results.append(("ood_values", np.array_equal(stk(p["ood_values"]), g["ood_values"])))
    results.append(("grinding_nonces", list(map(int, p["grinding_nonces"])) == list(g["grinding_nonces"])))
    results.append(("fold_grinding_nonces", list(map(int, p["fold_grinding_nonces"])) == list(g["fold_grinding_nonces"])))
    results.append(("initial_proof.opened_rows", rows_eq(p["initial_proof"]["opened_rows"], g["initial_proof"]["opened_rows"])))
    results.append(("initial_proof.merkle_proof", np.array_equal(p["initial_proof"]["merkle_proof"], g["initial_proof"]["merkle_proof"])))
    rp_ok = len(p["recursive_proofs"]) == len(g["recursive_proofs"])
    for pr, gr in zip(p["recursive_proofs"], g["recursive_proofs"]):
        rp_ok = rp_ok and rows_eq(pr["opened_rows"], gr["opened_rows"]) and np.array_equal(pr["merkle_proof"], gr["merkle_proof"])
    results.append(("recursive_proofs", rp_ok))
    results.append(("final_proof.yr", np.array_equal(np.asarray(p["final_proof"]["yr"]), g["final_proof"]["yr"])))
    results.append(("final_proof.opened_rows", rows_eq(p["final_proof"]["opened_rows"], g["final_proof"]["opened_rows"])))
    results.append(("final_proof.merkle_proof", np.array_equal(p["final_proof"]["merkle_proof"], g["final_proof"]["merkle_proof"])))
    return g, results


def main() -> int:
    print(f"device {frx.devices()[0]}")
    g, results = run()
    allok = True
    for nm, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}"); allok = allok and ok
    print(f"ligerito prove_flock_ligerito vs flock (log_n={g['log_n']}, "
          f"R={g['cfg']['recursive_steps']}): {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
