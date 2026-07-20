"""GPU 3-wide Keccak (keccak3) prover with the LIGERITO PCS, byte gate vs flock
keccak3::KeccakSetup::prove_fast — THE headline keccak path. Task #14, M3b.

Ingests dump_keccak3_ligerito (keccak3 R1CS + Ligerito config + walker probes +
full R1csProofLigerito), replays flock-zorch's prover on one shared challenger
(commit → bind → zerocheck → keccak3 walker lincheck → batched Ligerito open) and
byte-compares every field. keccak3's A_0/B_0 are empty stubs, so the circuit is
the procedural Keccak3LincheckCircuit (3 disjoint sub-keccak walks).
  W: walker probes  (Keccak3LincheckCircuit.fold_alpha_batched — the circuit gate)
  full Ligerito proof (commit/zerocheck/lincheck/recursive open)

Run (regen: cargo run --release --example dump_keccak3_ligerito -- 49 artifacts/keccak3_ligerito_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  FRX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/keccak3_ligerito_oracle_test.py
"""
import sys
from pathlib import Path

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402

from flock_zorch import zerocheck, lincheck, prover, ghash  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.lincheck.keccak3 import Keccak3LincheckCircuit  # noqa: E402

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


def _unpack(zp, m):
    zp = np.asarray(zp, np.uint64).reshape(-1, 2)
    bi = np.arange(64, dtype=np.uint64)
    lo = ((zp[:, 0:1] >> bi) & np.uint64(1)).astype(np.uint8)
    hi = ((zp[:, 1:2] >> bi) & np.uint64(1)).astype(np.uint8)
    return np.concatenate([lo, hi], axis=1).reshape(-1)[: 1 << m]


def load(golden: str = "keccak3_ligerito_golden.bin"):
    """Ingest a golden. `golden` names a file under `artifacts/`, so the same
    loader serves the m-variant dumps a size sweep needs."""
    rd = R((ART / golden).read_bytes())
    assert bytes(rd.take(8)) == b"FLKK3L01", "bad magic"
    meta = dict(m=rd.u(), k_log=rd.u(), k_skip=rd.u(), useful_bits=rd.u(), const_pin=rd.u(),
                lir=rd.u(), lbs=rd.u(), n_blocks_log=rd.u(), log_n=rd.u())
    cfg = dict(log_inv_rates=rd.uv(), recursive_steps=rd.u(), initial_log_msg_cols=rd.u(),
               initial_log_num_interleaved=rd.u(), initial_k=rd.u(), recursive_log_msg_cols=rd.uv(),
               recursive_ks=rd.uv(), queries=rd.uv(), grinding_bits=rd.uv(),
               fold_grinding_bits=rd.uv(), ood_samples=rd.uv())
    g = dict(meta=meta, cfg=cfg, stmt=bytes(rd.raw(32)), root=rd.raw(32),
             z=rd.fv(), a=rd.fv(), b=rd.fv())
    g["zlc"] = bytes(rd.raw(rd.u()))
    g["probes"] = [dict(alpha=rd.f(), eq=rd.fv(), comb=rd.fv()) for _ in range(rd.u())]
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
    k_log, k_skip = meta["k_log"], meta["k_skip"]; ir = k_log - k_skip   # inner_rest = 17 - 6 = 11
    results = []

    root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, g["z"])
    results.append(("commit root", np.array_equal(root, g["root"])))

    ch = Challenger(b"flock-keccak3-lig-v0")
    prover.bind_statement(ch, g["stmt"], root)
    a_bits, b_bits, c_bits = _unpack(g["a"], m), _unpack(g["b"], m), _unpack(g["z"], m)
    zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch)
    results.append(("zerocheck round1_ab", np.array_equal(zc.round1_ab, g["zc"]["r1ab"])))
    results.append(("zerocheck final_c", np.array_equal(zc.final_c_eval, g["zc"]["fc"])))

    circ = Keccak3LincheckCircuit()
    x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
    _lr, lc_zp, lc_claim, _zv = lincheck.prove(g["zlc"], None, None, x_ab, m, k_log, k_skip, ch=ch, capture=True, circuit=circ)
    results.append(("lincheck z_partial", np.array_equal(ghash.to_lanes(lc_zp), g["lc"]["zp"])))

    ab_full = fnp.concatenate([lc_claim.r_inner_rest, x_ab.x_outer], axis=0)
    c_full = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
    out = prover.open_batch_ligerito(cfg, g["z"], pdata, [ab_full, c_full], ch)

    for i in range(len(g["rs"])):
        results.append((f"open ring_switch[{i}]", np.array_equal(ghash.to_lanes(out.ring_switches[i]), g["rs"][i])))
    p, gl = out.ligerito, g["lig"]

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

    # Stage W: the keccak3 walker port — Keccak3LincheckCircuit.fold_alpha_batched (standalone)
    for i, pb in enumerate(g["probes"]):
        comb = ghash.from_ghash_host(circ.fold_alpha_batched(ghash.to_ghash(pb["alpha"]), ghash.to_ghash(pb["eq"])))
        results.append((f"walker probe {i} (fold_alpha_batched)", np.array_equal(comb, pb["comb"])))
    return m, results


def main() -> int:
    print(f"device {frx.devices()[0]}")
    m, results = run()
    allok = True
    for nm, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}"); allok = allok and ok
    print(f"keccak3 LIGERITO full prove (R1csProofLigerito) vs flock prove_fast (m={m}): "
          f"{'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
