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
