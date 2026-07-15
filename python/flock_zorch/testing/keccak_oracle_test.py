"""GPU Keccak-f[1600] R1CS prover byte gate vs flock (BaseFold). Task #14.

Keccak's A_0/B_0 are EMPTY stubs — the lincheck constraints live in the procedural
KeccakLincheckCircuit walker. So this gates stage by stage on ONE shared challenger:
  A: commit root        (ingested keccak witness)
  B: zerocheck          (real a=A·z, b=B·z, c=z; useful_bits padding)
  C: walker lincheck    (KeccakLincheckCircuit threaded through lincheck.prove,
                         const_pin=Z_CONST +β, inner_rest=k_log-k_skip=10)
  D: batched dual-claim open (ab from lincheck, c from zerocheck) — full reuse
  W: walker probes      (flock fold_alpha_batched samples — the standalone M1 gate)

Run (regen: cargo run --release --example dump_keccak -- 8 artifacts/keccak_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/keccak_oracle_test.py
"""
import sys
from pathlib import Path

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch import field, zerocheck, lincheck, prover  # noqa: E402
from flock_zorch.pcs import commit as pcs_commit  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.lincheck.keccak import KeccakLincheckCircuit  # noqa: E402

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


def _eq(name, got, want, results):
    g = np.asarray(got, np.uint64).reshape(-1, 2); w = np.asarray(want, np.uint64).reshape(-1, 2)
    results.append((name, g.shape == w.shape and np.array_equal(g, w)))


def _unpack(zp, m):
    zp = np.asarray(zp, np.uint64).reshape(-1, 2)
    bi = np.arange(64, dtype=np.uint64)
    lo = ((zp[:, 0:1] >> bi) & np.uint64(1)).astype(np.uint8)
    hi = ((zp[:, 1:2] >> bi) & np.uint64(1)).astype(np.uint8)
    return np.concatenate([lo, hi], axis=1).reshape(-1)[: 1 << m]


def load():
    rd = R((ART / "keccak_golden.bin").read_bytes())
    assert bytes(rd.take(8)) == b"FLKKEC01", "bad magic"
    meta = dict(m=rd.u(), k_log=rd.u(), k_skip=rd.u(), useful_bits=rd.u(), const_pin=rd.u(),
                lir=rd.u(), lbs=rd.u(), n_blocks_log=rd.u(), K=rd.u())
    g = dict(meta=meta, stmt=bytes(rd.raw(32)), root=rd.raw(32), z=rd.fv(), a=rd.fv(), b=rd.fv())
    g["zlc"] = bytes(rd.raw(rd.u()))
    g["probes"] = [dict(alpha=rd.f(), eq=rd.fv(), comb=rd.fv()) for _ in range(rd.u())]
    g["zc"] = dict(r1ab=rd.fv(), r1c=rd.fv(), mlv=rd.pair(), fa=rd.f(), fb=rd.f(), fc=rd.f())
    g["lc"] = dict(rounds=rd.pair(), zp=rd.fv())
    g["rs"] = [rd.fv() for _ in range(rd.u())]
    bf = dict(rm=rd.pair(), post_rb_root=rd.raw(32), rc=rd.hv(), fa=rd.f(), fb=rd.f(), fcw=rd.fv())
    nq = rd.u()
    bf["queries"] = [(rd.u(), rd.fv(), rd.fv(), [rd.fv() for _ in range(rd.u())]) for _ in range(nq)]
    bf["imp"] = rd.hv(); bf["prmp"] = rd.hv(); bf["emp"] = [rd.hv() for _ in range(rd.u())]
    g["bf"] = bf
    return g


def run():
    g = load(); meta = g["meta"]; m, lir, lbs = meta["m"], meta["lir"], meta["lbs"]
    results = []

    # Stage A: commit root on the ingested packed witness
    root, codeword, tree = pcs_commit.commit(g["z"], m, lir, lbs)
    _eq("commit root", root, g["root"], results)

    # Stage B: zerocheck on real a=A·z, b=B·z, c=z (useful_bits padding) — shared challenger
    ch = Challenger(b"flock-keccak-v0")
    prover.bind_statement(ch, g["stmt"], root)
    a_bits, b_bits, c_bits = _unpack(g["a"], m), _unpack(g["b"], m), _unpack(g["z"], m)
    zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch)
    _eq("zerocheck round1_ab", zc.round1_ab, g["zc"]["r1ab"], results)
    _eq("zerocheck round1_c", zc.round1_c, g["zc"]["r1c"], results)
    got_mlv = np.array([np.concatenate([a, b]) for a, b in zc.multilinear_rounds])
    want_mlv = np.array([np.concatenate([a, b]) for a, b in g["zc"]["mlv"]])
    _eq("zerocheck multilinear_rounds", got_mlv, want_mlv, results)
    _eq("zerocheck final_c", zc.final_c_eval, g["zc"]["fc"], results)

    # Stage C: walker lincheck — KeccakLincheckCircuit threaded through lincheck.prove
    ir = meta["k_log"] - meta["k_skip"]                     # inner_rest = 16 - 6 = 10
    circ = KeccakLincheckCircuit()
    x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
    lc_rounds, lc_zp, lc_claim, _zvp = lincheck.prove(
        g["zlc"], None, None, x_ab, m, meta["k_log"], meta["k_skip"], ch=ch, capture=True, circuit=circ)
    got_lcr = np.array([np.concatenate([a, b]) for a, b in lc_rounds]) if lc_rounds else np.zeros((0, 4), np.uint64)
    want_lcr = np.array([np.concatenate([a, b]) for a, b in g["lc"]["rounds"]]) if g["lc"]["rounds"] else np.zeros((0, 4), np.uint64)
    results.append((f"lincheck rounds (walker, inner_rest={ir})",
                    got_lcr.shape == want_lcr.shape and np.array_equal(got_lcr, want_lcr)))
    _eq("lincheck z_partial", lc_zp, g["lc"]["zp"], results)

    # Stage D: batched dual-claim open (ab from lincheck, c from zerocheck)
    k_code = (m - 7 - lbs) + lir
    ab_full = np.concatenate([lc_claim.r_inner_rest, x_ab.x_outer], axis=0)
    c_full = np.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
    out = prover.open_batch(g["z"], codeword, tree, [ab_full, c_full], k_code, lir, lbs, ch)
    for i in range(len(g["rs"])):
        _eq(f"open ring_switch[{i}]", out.ring_switches[i], g["rs"][i], results)
    bf = out.basefold; gbf = g["bf"]
    got_rm = np.array([np.concatenate([a, b]) for a, b in bf.round_messages])
    want_rm = np.array([np.concatenate([a, b]) for a, b in gbf["rm"]])
    _eq("open bf round_messages", got_rm, want_rm, results)
    _eq("open bf post_rb_commit", bf.post_row_batch_commit, gbf["post_rb_root"], results)
    rc = np.stack(bf.round_commitments) if len(bf.round_commitments) else np.zeros((0, 32), np.uint8)
    results.append(("open bf round_commitments", rc.shape == gbf["rc"].shape and np.array_equal(rc, gbf["rc"])))
    _eq("open bf final_a", bf.final_a, gbf["fa"], results)
    _eq("open bf final_b", bf.final_b, gbf["fb"], results)
    _eq("open bf final_codeword", bf.final_codeword, gbf["fcw"], results)
    q_ok = len(bf.queries) == len(gbf["queries"])
    for (gp, gil, gprl, gel), (pos, il, prl, el) in zip(gbf["queries"], bf.queries):
        q_ok = q_ok and pos == gp and np.array_equal(np.asarray(il, np.uint64).reshape(-1, 2), gil)
        q_ok = q_ok and np.array_equal(np.asarray(prl, np.uint64).reshape(-1, 2), gprl)
        q_ok = q_ok and len(el) == len(gel) and \
            all(np.array_equal(np.asarray(a, np.uint64).reshape(-1, 2), b) for a, b in zip(el, gel))
    results.append(("open bf queries", q_ok))
    results.append(("open bf initial_multi_proof",
                    np.array_equal(np.asarray(bf.initial_multi_proof), gbf["imp"])))
    results.append(("open bf post_rb_multi_proof",
                    np.array_equal(np.asarray(bf.post_row_batch_multi_proof), gbf["prmp"])))
    emp_ok = len(bf.epoch_multi_proofs) == len(gbf["emp"]) and \
        all(np.array_equal(np.asarray(a), b) for a, b in zip(bf.epoch_multi_proofs, gbf["emp"]))
    results.append(("open bf epoch_multi_proofs", emp_ok))

    # Stage W: the M1 walker port — KeccakLincheckCircuit.fold_alpha_batched (standalone)
    for i, p in enumerate(g["probes"]):
        comb = circ.fold_alpha_batched(p["alpha"], p["eq"])
        results.append((f"walker probe {i} (fold_alpha_batched)", np.array_equal(comb, p["comb"])))
    return m, results


def main() -> int:
    print(f"device {frx.devices()[0]}")
    m, results = run()
    allok = True
    for nm, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}"); allok = allok and ok
    print(f"keccak FULL prove (commit+zerocheck+walker lincheck+batched open) vs flock BaseFold "
          f"(m={m}): {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
