"""GPU Ligerito recursive PCS byte gate vs flock (driver-isolated).

Ingests the dump_ligerito golden (config + synthetic f/b/target + L0 commit + full
LigeritoProof, via flock's pub recursive_prover_with_basis) and byte-compares the
flock-zorch Ligerito port stage by stage. Built up across milestones:
  M0: ligero_commit on the L0 poly == initial_root (+ L0 codeword matches)
  [M1: SumcheckProver lane-folds; M2: L0 open/induce; ... as the port progresses]

Run (regen golden: cargo run --release --example dump_ligerito -- 15 artifacts/ligerito_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH=python:/home/jooman/fractalyze/zorch <venv> \
      python/flock_zorch/testing/ligerito_oracle_test.py
"""
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from flock_zorch import field, ligerito  # noqa: E402
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


def run(mul):
    _, g = load()
    cfg = g["cfg"]
    results = []

    # M0: ligero_commit on the L0 poly (f) == the proof's initial_root + L0 codeword
    mat, tree = ligerito.ligero_commit(
        g["f"], cfg["initial_log_msg_cols"], cfg["initial_log_num_interleaved"],
        cfg["log_inv_rates"][0], mul=mul)
    results.append(("ligero_commit L0 codeword", np.array_equal(mat, g["l0_codeword"])))
    results.append(("ligero_commit L0 root == initial_root", np.array_equal(tree[-1], g["initial_root"])))

    # M1: lane folds (initial_k) + commit f^1 — shared challenger
    log_n = g["log_n"]; initial_k = cfg["initial_k"]
    ch = Challenger(b"flock-ligerito-test")
    ch.observe_label(b"flock-ligerito-basis-v0")
    ch.observe_f128(g["target"])
    ch.observe_bytes(bytes(g["initial_root"]))
    sc, start = ligerito.SumcheckProver.new(g["f"], g["b"], g["target"], mul)
    ch.observe_f128(start[0]); ch.observe_f128(start[1])
    fold_nonces = []
    fb0 = cfg["fold_grinding_bits"][0]
    for j in range(initial_k):
        bits = max(fb0 - j, 0)
        if bits > 0:
            fold_nonces.append(ch.grind_pow(bits))
        r = ch.sample_f128()
        msg = sc.fold(r)
        ch.observe_f128(msg[0]); ch.observe_f128(msg[1])
    # commit f^1
    n1 = log_n - initial_k
    lni1 = cfg["recursive_ks"][0]
    mat1, tree1 = ligerito.ligero_commit(sc.f, n1 - lni1, lni1, cfg["log_inv_rates"][1], mul=mul)

    # gate the first initial_k+1 sumcheck messages + L1 root + lane fold nonces
    got_tr = np.array([np.concatenate([a, b]) for a, b in sc.transcript])
    want_tr = np.array([np.concatenate([a, b]) for a, b in g["sumcheck_transcript"][:initial_k + 1]])
    results.append(("M1 lane-fold sumcheck msgs", np.array_equal(got_tr, want_tr)))
    results.append(("M1 recursive_roots[0] (f^1 commit)", np.array_equal(tree1[-1], g["recursive_roots"][0])))
    results.append(("M1 lane fold_grinding_nonces", fold_nonces == g["fold_grinding_nonces"][:len(fold_nonces)]))
    return g, results


def main() -> int:
    print(f"device {jax.devices()[0]} | mul {'clmad' if MUL is not field.mul else 'software'}")
    g, results = run(MUL)
    allok = True
    for nm, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}"); allok = allok and ok
    print(f"ligerito M0 (ligero_commit) vs flock (log_n={g['log_n']}, R={g['cfg']['recursive_steps']}): "
          f"{'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
