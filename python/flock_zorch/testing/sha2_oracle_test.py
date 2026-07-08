"""GPU SHA-256 R1CS prover byte gate vs flock `prove` (BaseFold), real sha2 R1CS.

Ingests flock's real sha2 instance (dump_sha2.rs: Sha256HybridSetup, K_LOG=15,
const_pin=31400, USEFUL_BITS=31401, inner_rest=9) — witness gen + the R1CS are
host setup, not the GPU target — and replays flock-zorch's prover phases on ONE
shared challenger, byte-comparing each stage. Stages built up incrementally:
  A: commit root          (PCS commit on the real packed witness)
  B: zerocheck proof+claim (real a=A·z, b=B·z, c=z; useful_bits padding)
  [C: lincheck CSC fold, D: open — added as the sha2 port progresses]

Run (regen golden: cargo run --release --example dump_sha2 -- 8 artifacts/sha2_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/sha2_oracle_test.py
"""
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from flock_zorch import field, zerocheck, lincheck, prover  # noqa: E402
from flock_zorch.pcs import commit as pcs_commit  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402

ART = Path(__file__).resolve().parents[3] / "artifacts"


class R:
    def __init__(self, buf): self.b = buf; self.o = 0
    def take(self, n): v = self.b[self.o:self.o + n]; self.o += n; return v
    def u(self): return int.from_bytes(self.take(8), "little")
    def f(self): return np.frombuffer(self.take(16), np.uint64).copy()
    def fv(self): n = self.u(); return np.frombuffer(self.take(16 * n), np.uint64).reshape(n, 2).copy()
    def pair(self): n = self.u(); return [(self.f(), self.f()) for _ in range(n)]
    def raw(self, n): return np.frombuffer(self.take(n), np.uint8).copy()
    def hv(self): n = self.u(); return np.frombuffer(self.take(32 * n), np.uint8).reshape(n, 32).copy()
    def rows(self):
        nr = self.u(); out = []
        for _ in range(nr):
            ln = self.u(); out.append(np.frombuffer(self.take(4 * ln), np.uint32).copy())
        return out


def _unpack(z_packed, m):
    zp = np.asarray(z_packed, np.uint64).reshape(-1, 2)
    bi = np.arange(64, dtype=np.uint64)
    lo = ((zp[:, 0:1] >> bi) & np.uint64(1)).astype(np.uint8)
    hi = ((zp[:, 1:2] >> bi) & np.uint64(1)).astype(np.uint8)
    return np.concatenate([lo, hi], axis=1).reshape(-1)[: 1 << m]


def _eq(name, got, want, results):
    g = np.asarray(got, np.uint64).reshape(-1, 2); w = np.asarray(want, np.uint64).reshape(-1, 2)
    results.append((name, g.shape == w.shape and np.array_equal(g, w)))


def load():
    rd = R((ART / "sha2_golden.bin").read_bytes())
    assert bytes(rd.take(8)) == b"FLKSHA01", "bad magic"
    meta = dict(m=rd.u(), k_log=rd.u(), k_skip=rd.u(), useful_bits=rd.u(),
                const_pin=rd.u(), lir=rd.u(), lbs=rd.u(), n_blocks_log=rd.u())
    g = dict(meta=meta, stmt=bytes(rd.raw(32)), root=rd.raw(32),
             z=rd.fv(), a=rd.fv(), b=rd.fv())
    zlc_n = rd.u(); g["zlc"] = bytes(rd.raw(zlc_n))
    g["a0_rows"] = rd.rows(); g["b0_rows"] = rd.rows()
    g["zc"] = dict(r1ab=rd.fv(), r1c=rd.fv(), mlv=rd.pair(), fa=rd.f(), fb=rd.f(), fc=rd.f())
    g["lc"] = dict(rounds=rd.pair(), zp=rd.fv())
    g["rs"] = [rd.fv() for _ in range(rd.u())]
    bf = dict(rm=rd.pair(), post_rb_root=rd.raw(32), rc=rd.hv(), fa=rd.f(), fb=rd.f(), fcw=rd.fv())
    nq = rd.u()
    bf["queries"] = [(rd.u(), rd.fv(), rd.fv(), [rd.fv() for _ in range(rd.u())]) for _ in range(nq)]
    bf["imp"] = rd.hv(); bf["prmp"] = rd.hv(); bf["emp"] = [rd.hv() for _ in range(rd.u())]
    g["bf"] = bf
    return rd, g


def run():
    _, g = load()
    m = g["meta"]["m"]; lir = g["meta"]["lir"]; lbs = g["meta"]["lbs"]
    results = []

    # Stage A: commit root on the real packed witness
    root, codeword, tree = pcs_commit.commit(g["z"], m, lir, lbs)
    _eq("commit root", root, g["root"], results)

    # Stage B: zerocheck on real a=A·z, b=B·z, c=z (useful_bits padding) — shared challenger
    ch = Challenger(b"flock-sha2-v0")
    prover.bind_statement(ch, g["stmt"], root)
    a_bits, b_bits, c_bits = g["a"], g["b"], g["z"]  # packed F128 — witness_to_rows unpacks on device
    zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch)
    _eq("zc round1_ab", zc.round1_ab, g["zc"]["r1ab"], results)
    _eq("zc round1_c", zc.round1_c, g["zc"]["r1c"], results)
    got_mlv = np.array([np.concatenate([a, b]) for a, b in zc.multilinear_rounds])
    want_mlv = np.array([np.concatenate([a, b]) for a, b in g["zc"]["mlv"]])
    _eq("zc multilinear_rounds", got_mlv, want_mlv, results)
    _eq("zc final_a", zc.final_a_eval, g["zc"]["fa"], results)
    _eq("zc final_b", zc.final_b_eval, g["zc"]["fb"], results)
    _eq("zc final_c", zc.final_c_eval, g["zc"]["fc"], results)

    # Stage C: CSC sparse lincheck (k=32768, const_pin=31400, inner_rest=9) — shared challenger
    k = 1 << g["meta"]["k_log"]
    csc = lincheck.CscCircuit(g["a0_rows"], g["b0_rows"], k, const_pin=g["meta"]["const_pin"])
    ir = g["meta"]["k_log"] - g["meta"]["k_skip"]
    x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
    lc_rounds, lc_zp, lc_claim, _zvp = lincheck.prove(
        g["zlc"], None, None, x_ab, m, g["meta"]["k_log"], g["meta"]["k_skip"], ch=ch, capture=True, circuit=csc)
    got_lcr = np.array([np.concatenate([a, b]) for a, b in lc_rounds]) if lc_rounds else np.zeros((0, 4), np.uint64)
    want_lcr = np.array([np.concatenate([a, b]) for a, b in g["lc"]["rounds"]]) if g["lc"]["rounds"] else np.zeros((0, 4), np.uint64)
    results.append(("lc rounds (CSC, inner_rest=9)", got_lcr.shape == want_lcr.shape and np.array_equal(got_lcr, want_lcr)))
    _eq("lc z_partial", lc_zp, g["lc"]["zp"], results)

    # Stage D: batched dual-claim open (ab from lincheck, c from zerocheck)
    ab_full = np.concatenate([lc_claim.r_inner_rest, x_ab.x_outer], axis=0)
    c_full = np.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
    out = prover.open_batch(g["z"], codeword, tree, [ab_full, c_full], (m - 7 - lbs) + lir,
                            lir, lbs, ch)
    for i in range(2):
        _eq(f"open ring_switch[{i}]", out.ring_switches[i], g["rs"][i], results)
    bf = out.basefold; gbf = g["bf"]
    got_rm = np.array([np.concatenate([a, b]) for a, b in bf["round_messages"]])
    want_rm = np.array([np.concatenate([a, b]) for a, b in gbf["rm"]])
    _eq("open bf round_messages", got_rm, want_rm, results)
    _eq("open bf post_rb_commit", bf["post_row_batch_commit"], gbf["post_rb_root"], results)
    rc = np.stack(bf["round_commitments"]) if len(bf["round_commitments"]) else np.zeros((0, 32), np.uint8)
    results.append(("open bf round_commitments", rc.shape == gbf["rc"].shape and np.array_equal(rc, gbf["rc"])))
    _eq("open bf final_a", bf["final_a"], gbf["fa"], results)
    _eq("open bf final_b", bf["final_b"], gbf["fb"], results)
    _eq("open bf final_codeword", bf["final_codeword"], gbf["fcw"], results)
    q_ok = len(bf["queries"]) == len(gbf["queries"])
    for (gp, gil, gprl, gel), (pos, il, prl, el) in zip(gbf["queries"], bf["queries"]):
        q_ok = q_ok and pos == gp and np.array_equal(il, gil) and np.array_equal(prl, gprl)
        q_ok = q_ok and len(el) == len(gel) and all(np.array_equal(a, b) for a, b in zip(el, gel))
    results.append(("open bf queries", q_ok))
    _eq("open bf initial_multi_proof", bf["initial_multi_proof"], gbf["imp"], results)
    _eq("open bf post_rb_multi_proof", bf["post_row_batch_multi_proof"], gbf["prmp"], results)
    emp_ok = len(bf["epoch_multi_proofs"]) == len(gbf["emp"]) and \
        all(np.array_equal(a, b) for a, b in zip(bf["epoch_multi_proofs"], gbf["emp"]))
    results.append(("open bf epoch_multi_proofs", emp_ok))
    return m, results


def main() -> int:
    print(f"device {jax.devices()[0]}")
    m, results = run()
    allok = True
    for nm, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}"); allok = allok and ok
    print(f"sha2 FULL prove (commit+zerocheck+CSC lincheck+batched open) vs flock BaseFold "
          f"(m={m}): {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
