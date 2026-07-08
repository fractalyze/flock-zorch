"""GPU SHA-256 prover with the LIGERITO PCS, byte gate vs flock prove_ligerito —
the HEADLINE sha256 path (M7).

Ingests dump_sha2_ligerito (real sha2 R1CS + Ligerito config + full
R1csProofLigerito), replays flock-zorch's prover on one shared challenger
(commit → bind → zerocheck → CSC lincheck → batched Ligerito open) and
byte-compares every field of the R1csProofLigerito.

Run (regen: cargo run --release --example dump_sha2_ligerito -- 128 artifacts/sha2_ligerito_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/sha2_ligerito_oracle_test.py
"""
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from flock_zorch import zorch_ligerito, zerocheck, lincheck, prover  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402

ART = Path(__file__).resolve().parents[3] / "artifacts"


class R:
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


def _unpack(zp, m):
    zp = np.asarray(zp, np.uint64).reshape(-1, 2)
    bi = np.arange(64, dtype=np.uint64)
    lo = ((zp[:, 0:1] >> bi) & np.uint64(1)).astype(np.uint8)
    hi = ((zp[:, 1:2] >> bi) & np.uint64(1)).astype(np.uint8)
    return np.concatenate([lo, hi], axis=1).reshape(-1)[: 1 << m]


def load():
    rd = R((ART / "sha2_ligerito_golden.bin").read_bytes())
    assert bytes(rd.take(8)) == b"FLKSL_01", "bad magic"
    meta = dict(m=rd.u(), k_log=rd.u(), k_skip=rd.u(), useful_bits=rd.u(), const_pin=rd.u(),
                lir=rd.u(), lbs=rd.u(), n_blocks_log=rd.u(), log_n=rd.u())
    cfg = dict(log_inv_rates=rd.uv(), recursive_steps=rd.u(), initial_log_msg_cols=rd.u(),
               initial_log_num_interleaved=rd.u(), initial_k=rd.u(), recursive_log_msg_cols=rd.uv(),
               recursive_ks=rd.uv(), queries=rd.uv(), grinding_bits=rd.uv(),
               fold_grinding_bits=rd.uv(), ood_samples=rd.uv())
    g = dict(meta=meta, cfg=cfg, stmt=bytes(rd.raw(32)), root=rd.raw(32),
             z=rd.fv(), a=rd.fv(), b=rd.fv())
    g["zlc"] = bytes(rd.raw(rd.u()))
    g["a0_rows"] = rd.rowsu(); g["b0_rows"] = rd.rowsu()
    g["zc"] = dict(r1ab=rd.fv(), r1c=rd.fv(), mlv=rd.pair(), fa=rd.f(), fb=rd.f(), fc=rd.f())
    g["lc"] = dict(rounds=rd.pair(), zp=rd.fv())
    g["rs"] = [rd.fv() for _ in range(rd.u())]
    lig = dict(initial_root=rd.raw(32))
    lig["initial_proof"] = dict(opened_rows=rd.rowsf(), merkle_proof=rd.hv())
    lig["recursive_roots"] = rd.hv()
    nrp = rd.u(); lig["recursive_proofs"] = [dict(opened_rows=rd.rowsf(), merkle_proof=rd.hv()) for _ in range(nrp)]
    lig["final_proof"] = dict(yr=rd.fv(), opened_rows=rd.rowsf(), merkle_proof=rd.hv())
    lig["sumcheck_transcript"] = rd.pair()
    lig["grinding_nonces"] = rd.u64v(); lig["ood_values"] = rd.fv(); lig["fold_grinding_nonces"] = rd.u64v()
    g["lig"] = lig
    return g


def run():
    g = load(); meta = g["meta"]; cfg = g["cfg"]
    m = meta["m"]
    k_log, k_skip = meta["k_log"], meta["k_skip"]; ir = k_log - k_skip
    results = []

    root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, g["z"])
    results.append(("commit root", np.array_equal(root, g["root"])))

    ch = Challenger(b"flock-sha2-lig-v0")
    prover.bind_statement(ch, g["stmt"], root)
    a_bits, b_bits, c_bits = g["a"], g["b"], g["z"]  # packed F128 — witness_to_rows unpacks on device
    zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch)
    results.append(("zerocheck round1_ab", np.array_equal(zc.round1_ab, g["zc"]["r1ab"])))
    results.append(("zerocheck final_c", np.array_equal(zc.final_c_eval, g["zc"]["fc"])))

    csc = lincheck.CscCircuit(g["a0_rows"], g["b0_rows"], 1 << k_log, const_pin=meta["const_pin"])
    x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
    _lr, lc_zp, lc_claim, _zv = lincheck.prove(g["zlc"], None, None, x_ab, m, k_log, k_skip, ch=ch, capture=True, circuit=csc)
    results.append(("lincheck z_partial", np.array_equal(lc_zp, g["lc"]["zp"])))

    ab_full = np.concatenate([lc_claim.r_inner_rest, x_ab.x_outer], axis=0)
    c_full = np.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
    out = prover.open_batch_ligerito(cfg, g["z"], pdata, [ab_full, c_full], ch)

    for i in range(len(g["rs"])):
        results.append((f"open ring_switch[{i}]", np.array_equal(out["ring_switches"][i], g["rs"][i])))
    p, gl = out["ligerito"], g["lig"]

    def pairs(t): return np.array([np.concatenate([a, b]) for a, b in t]) if t else np.zeros((0, 4), np.uint64)
    def rows_eq(a, b): return len(a) == len(b) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
    def stk(v): return np.stack([np.asarray(x).reshape(2) for x in v]) if len(v) else np.zeros((0, 2), np.uint64)

    results.append(("lig initial_root", np.array_equal(p["initial_root"], gl["initial_root"])))
    results.append(("lig sumcheck_transcript", np.array_equal(pairs(p["sumcheck_transcript"]), pairs(gl["sumcheck_transcript"]))))
    results.append(("lig recursive_roots", np.array_equal(np.asarray(p["recursive_roots"]), gl["recursive_roots"])))
    results.append(("lig ood_values", np.array_equal(stk(p["ood_values"]), gl["ood_values"])))
    results.append(("lig grinding_nonces", list(map(int, p["grinding_nonces"])) == list(gl["grinding_nonces"])))
    results.append(("lig fold_grinding_nonces", list(map(int, p["fold_grinding_nonces"])) == list(gl["fold_grinding_nonces"])))
    results.append(("lig initial_proof.opened_rows", rows_eq(p["initial_proof"]["opened_rows"], gl["initial_proof"]["opened_rows"])))
    results.append(("lig initial_proof.merkle_proof", np.array_equal(p["initial_proof"]["merkle_proof"], gl["initial_proof"]["merkle_proof"])))
    rp_ok = len(p["recursive_proofs"]) == len(gl["recursive_proofs"])
    for pr, gr in zip(p["recursive_proofs"], gl["recursive_proofs"]):
        rp_ok = rp_ok and rows_eq(pr["opened_rows"], gr["opened_rows"]) and np.array_equal(pr["merkle_proof"], gr["merkle_proof"])
    results.append(("lig recursive_proofs", rp_ok))
    results.append(("lig final_proof.yr", np.array_equal(np.asarray(p["final_proof"]["yr"]), gl["final_proof"]["yr"])))
    results.append(("lig final_proof.opened_rows", rows_eq(p["final_proof"]["opened_rows"], gl["final_proof"]["opened_rows"])))
    results.append(("lig final_proof.merkle_proof", np.array_equal(p["final_proof"]["merkle_proof"], gl["final_proof"]["merkle_proof"])))
    return m, results


def main() -> int:
    print(f"device {jax.devices()[0]}")
    m, results = run()
    allok = True
    for nm, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}"); allok = allok and ok
    print(f"sha2 LIGERITO full prove (R1csProofLigerito) vs flock prove_ligerito (m={m}): "
          f"{'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
