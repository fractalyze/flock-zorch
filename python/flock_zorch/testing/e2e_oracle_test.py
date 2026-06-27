"""End-to-end fused-prover byte gate vs flock `prover::prove` (identity R1CS).

Reads artifacts/e2e_golden.bin (dump_e2e.rs: flock prove on identity_r1cs
m,k_log=6,k_skip=6,ub=64) and replays flock-zorch's fused prover on ONE shared
challenger, byte-comparing each stage: commit root, zerocheck proof+claim,
lincheck proof+claim, the ab/c z-claims, and the batched PCS open.

Run (regen golden first: cargo run --release --example dump_e2e -- 13 artifacts/e2e_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH=python:/home/jooman/fractalyze/zorch <venv> \
      python/flock_zorch/testing/e2e_oracle_test.py
"""
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from flock_zorch import field, pcs_commit, zerocheck, lincheck, prover, ring_switch  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402

ART = Path(__file__).resolve().parents[3] / "artifacts"

try:
    from flock_zorch import field_clmad
    MUL = field_clmad.mul if field_clmad.available() else field.mul
except Exception:  # noqa: BLE001
    MUL = field.mul


class R:
    def __init__(self, buf): self.b = buf; self.o = 0
    def take(self, n): v = self.b[self.o:self.o + n]; self.o += n; return v
    def u(self): return int.from_bytes(self.take(8), "little")
    def f(self): return np.frombuffer(self.take(16), np.uint64).copy()              # [2]
    def fv(self): n = self.u(); return np.frombuffer(self.take(16 * n), np.uint64).reshape(n, 2).copy()
    def pair(self): n = self.u(); return [(self.f(), self.f()) for _ in range(n)]
    def hv(self): n = self.u(); return np.frombuffer(self.take(32 * n), np.uint8).reshape(n, 32).copy()
    def raw(self, n): return np.frombuffer(self.take(n), np.uint8).copy()


def _eq(name, got, want, results):
    ok = np.array_equal(np.asarray(got, np.uint64).reshape(-1, 2) if np.asarray(got).size else np.asarray(got),
                        np.asarray(want, np.uint64).reshape(-1, 2) if np.asarray(want).size else np.asarray(want))
    results.append((name, ok))
    return ok


def _unpack(z_packed, m):
    zp = np.asarray(z_packed, np.uint64).reshape(-1, 2)
    bitidx = np.arange(64, dtype=np.uint64)
    lo = ((zp[:, 0:1] >> bitidx) & np.uint64(1)).astype(np.uint8)
    hi = ((zp[:, 1:2] >> bitidx) & np.uint64(1)).astype(np.uint8)
    return np.concatenate([lo, hi], axis=1).reshape(-1)[: 1 << m]


def run(mul):
    rd = R((ART / "e2e_golden.bin").read_bytes())
    assert bytes(rd.take(8)) == b"FLKE2E01", "bad magic"
    m, k_log, k_skip, ub = rd.u(), rd.u(), rd.u(), rd.u()
    stmt = bytes(rd.raw(32))
    z_packed = rd.fv()
    zlc = bytes(rd.raw(rd.u()))
    g_root = rd.raw(32)
    g_a = rd.fv(); g_b = rd.fv()
    g_zc = dict(r1ab=rd.fv(), r1c=rd.fv(), mlv=rd.pair(), fa=rd.f(), fb=rd.f(), fc=rd.f())
    g_zcl = dict(z=rd.f(), mlv=rd.fv(), rrest=rd.fv(), ae=rd.f(), be=rd.f(), ce=rd.f())
    g_shvc = rd.fv()
    g_lc = dict(rounds=rd.pair(), zp=rd.fv())
    g_lcl = dict(ris=rd.f(), rir=rd.fv(), w=rd.f())
    g_zvp = rd.fv()
    g_ab = dict(zs=rd.f(), xir=rd.fv(), xo=rd.fv(), v=rd.f())
    g_c = dict(zs=rd.f(), xir=rd.fv(), xo=rd.fv(), v=rd.f())
    # pcs_open golden: ring_switches + basefold
    g_rs = [rd.fv() for _ in range(rd.u())]
    g_bf = dict(rm=rd.pair(), post_rb_root=rd.raw(32), rc=rd.hv(), fa=rd.f(), fb=rd.f(), fcw=rd.fv())
    n_q = rd.u()
    g_bf["queries"] = [(rd.u(), rd.fv(), rd.fv(), [rd.fv() for _ in range(rd.u())]) for _ in range(n_q)]
    g_bf["imp"] = rd.hv(); g_bf["prmp"] = rd.hv()
    g_bf["emp"] = [rd.hv() for _ in range(rd.u())]

    results = []
    LIR, LBS = 1, 5

    # ---- Stage A: commit root ----
    root, codeword, tree = pcs_commit.commit(z_packed, m, LIR, LBS, mul=mul)
    _eq("commit root", root, g_root, results)

    # ---- shared challenger + bind_statement ----
    ch = Challenger(b"flock-test-v0")
    prover.bind_statement(ch, stmt, root)

    # ---- Stage B: zerocheck (a=b=c=z for identity) ----
    bits = _unpack(z_packed, m)
    _eq("a==z", g_a, z_packed, results)  # confirm identity witness in golden
    zc = zerocheck.prove_packed(bits, bits, bits, m, mul=mul, ch=ch)
    _eq("zc round1_ab", zc["round1_ab"], g_zc["r1ab"], results)
    _eq("zc round1_c", zc["round1_c"], g_zc["r1c"], results)
    got_mlv = np.array([np.concatenate([a, b]) for a, b in zc["multilinear_rounds"]])
    want_mlv = np.array([np.concatenate([a, b]) for a, b in g_zc["mlv"]])
    _eq("zc multilinear_rounds", got_mlv, want_mlv, results)
    _eq("zc final_a", zc["final_a_eval"], g_zc["fa"], results)
    _eq("zc final_b", zc["final_b_eval"], g_zc["fb"], results)
    _eq("zc final_c", zc["final_c_eval"], g_zc["fc"], results)
    _eq("zc claim z", zc["z"], g_zcl["z"], results)
    _eq("zc claim mlv_challenges", zc["mlv_challenges"], g_zcl["mlv"], results)
    _eq("zc claim r_rest", zc["r_rest"], g_zcl["rrest"], results)

    # ---- Stage C: lincheck (identity A0/B0, capture) ----
    k = 1 << k_log
    a0 = np.eye(k, dtype=np.uint64)
    b0 = np.eye(k, dtype=np.uint64)
    inner_rest = k_log - k_skip
    x_ab = {"z_skip": zc["z"],
            "x_inner_rest": zc["mlv_challenges"][:inner_rest],
            "x_outer": zc["mlv_challenges"][inner_rest:]}
    lc_rounds, lc_zp, lc_claim, z_vec_pre = lincheck.prove(
        zlc, a0, b0, x_ab, m, k_log, k_skip, mul=mul, ch=ch, capture=True)
    got_lcr = np.array([np.concatenate([a, b]) for a, b in lc_rounds]) if lc_rounds else np.zeros((0, 4), np.uint64)
    want_lcr = np.array([np.concatenate([a, b]) for a, b in g_lc["rounds"]]) if g_lc["rounds"] else np.zeros((0, 4), np.uint64)
    _eq("lc rounds", got_lcr, want_lcr, results)
    _eq("lc z_partial", lc_zp, g_lc["zp"], results)
    _eq("lc claim r_inner_skip", lc_claim["r_inner_skip"], g_lcl["ris"], results)
    _eq("lc claim r_inner_rest", lc_claim["r_inner_rest"], g_lcl["rir"], results)
    _eq("lc claim w", lc_claim["w"], g_lcl["w"], results)
    _eq("lc z_vec_pre", z_vec_pre, g_zvp, results)

    # ---- Stage D: ab / c z-claims ----
    ab_pt = dict(zs=lc_claim["r_inner_skip"], xir=lc_claim["r_inner_rest"], xo=x_ab["x_outer"], v=lc_claim["w"])
    c_pt = dict(zs=zc["z"], xir=zc["r_rest"][:inner_rest], xo=zc["r_rest"][inner_rest:], v=zc["final_c_eval"])
    _eq("ab.z_skip", ab_pt["zs"], g_ab["zs"], results)
    _eq("ab.x_outer", ab_pt["xo"], g_ab["xo"], results)
    _eq("ab.value", ab_pt["v"], g_ab["v"], results)
    _eq("c.z_skip", c_pt["zs"], g_c["zs"], results)
    _eq("c.x_outer", c_pt["xo"], g_c["xo"], results)
    _eq("c.value", c_pt["v"], g_c["v"], results)

    # ---- Stage E: batched dual-claim PCS open ----
    ab_full = np.concatenate([ab_pt["xir"], ab_pt["xo"]], axis=0)
    c_full = np.concatenate([c_pt["xir"], c_pt["xo"]], axis=0)
    out = prover.open_batch(z_packed, codeword, tree, [ab_full, c_full], (m - 7 - LBS) + LIR,
                            LIR, LBS, ch, mul=mul)
    # ring_switches (s_hat_v per claim)
    for i in range(2):
        _eq(f"open ring_switch[{i}].s_hat_v", out["ring_switches"][i], g_rs[i], results)
    bf = out["basefold"]
    got_rm = np.array([np.concatenate([a, b]) for a, b in bf["round_messages"]])
    want_rm = np.array([np.concatenate([a, b]) for a, b in g_bf["rm"]])
    _eq("open bf round_messages", got_rm, want_rm, results)
    _eq("open bf post_rb_commit", bf["post_row_batch_commit"], g_bf["post_rb_root"], results)
    rc = np.stack(bf["round_commitments"]) if len(bf["round_commitments"]) else np.zeros((0, 32), np.uint8)
    results.append(("open bf round_commitments", rc.shape == g_bf["rc"].shape and np.array_equal(rc, g_bf["rc"])))
    _eq("open bf final_a", bf["final_a"], g_bf["fa"], results)
    _eq("open bf final_b", bf["final_b"], g_bf["fb"], results)
    _eq("open bf final_codeword", bf["final_codeword"], g_bf["fcw"], results)
    # queries
    q_ok = len(bf["queries"]) == len(g_bf["queries"])
    for (gp, gil, gprl, gel), (pos, il, prl, el) in zip(g_bf["queries"], bf["queries"]):
        q_ok = q_ok and pos == gp and np.array_equal(il, gil) and np.array_equal(prl, gprl)
        q_ok = q_ok and len(el) == len(gel) and all(np.array_equal(a, b) for a, b in zip(el, gel))
    results.append(("open bf queries", q_ok))
    _eq("open bf initial_multi_proof", bf["initial_multi_proof"], g_bf["imp"], results)
    _eq("open bf post_rb_multi_proof", bf["post_row_batch_multi_proof"], g_bf["prmp"], results)
    emp_ok = len(bf["epoch_multi_proofs"]) == len(g_bf["emp"]) and \
        all(np.array_equal(a, b) for a, b in zip(bf["epoch_multi_proofs"], g_bf["emp"]))
    results.append(("open bf epoch_multi_proofs", emp_ok))

    return m, results


def main() -> int:
    name = "clmad" if MUL is not field.mul else "software"
    print(f"device {jax.devices()[0]} | mul {name}")
    m, results = run(MUL)
    allok = True
    for nm, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}")
        allok = allok and ok
    print(f"e2e full prove byte-match vs flock prove (identity m={m}, stages A-E): {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
