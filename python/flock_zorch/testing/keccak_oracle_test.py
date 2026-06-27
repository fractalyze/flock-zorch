"""GPU Keccak-f[1600] R1CS prover byte gate vs flock (BaseFold). Task #14.

Keccak's A_0/B_0 are EMPTY stubs — the lincheck constraints live in the procedural
KeccakLincheckCircuit walker. So this gates stage by stage:
  A: commit root        (ingested keccak witness)
  B: zerocheck          (real a=A·z, b=B·z, c=z; useful_bits padding)
  W: walker probes      (flock fold_alpha_batched samples — the gate for the M1
                         Python walker port; checked once keccak_lincheck lands)
  [C: walker lincheck, D: open — added as the port progresses]

Run (regen: cargo run --release --example dump_keccak -- 8 artifacts/keccak_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH=python:/home/jooman/fractalyze/zorch <venv> \
      python/flock_zorch/testing/keccak_oracle_test.py
"""
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from flock_zorch import field, pcs_commit, zerocheck, prover  # noqa: E402
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
    def f(self): return np.frombuffer(self.take(16), np.uint64).copy()
    def fv(self): n = self.u(); return np.frombuffer(self.take(16 * n), np.uint64).reshape(n, 2).copy()
    def pair(self): n = self.u(); return [(self.f(), self.f()) for _ in range(n)]
    def raw(self, n): return np.frombuffer(self.take(n), np.uint8).copy()


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
    return g


def run(mul):
    g = load(); meta = g["meta"]; m, lir, lbs = meta["m"], meta["lir"], meta["lbs"]
    results = []
    root, codeword, tree = pcs_commit.commit(g["z"], m, lir, lbs, mul=mul)
    results.append(("commit root", np.array_equal(root, g["root"])))

    ch = Challenger(b"flock-keccak-v0")
    prover.bind_statement(ch, g["stmt"], root)
    a_bits, b_bits, c_bits = _unpack(g["a"], m), _unpack(g["b"], m), _unpack(g["z"], m)
    zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, mul=mul, ch=ch)
    results.append(("zerocheck round1_ab", np.array_equal(zc["round1_ab"], g["zc"]["r1ab"])))
    results.append(("zerocheck round1_c", np.array_equal(zc["round1_c"], g["zc"]["r1c"])))
    gm = np.array([np.concatenate([a, b]) for a, b in zc["multilinear_rounds"]])
    wm = np.array([np.concatenate([a, b]) for a, b in g["zc"]["mlv"]])
    results.append(("zerocheck multilinear_rounds", np.array_equal(gm, wm)))
    results.append(("zerocheck final_c", np.array_equal(zc["final_c_eval"], g["zc"]["fc"])))
    print(f"  (walker probes available for M1: {len(g['probes'])}, eq len {g['probes'][0]['eq'].shape[0]})")
    return m, results


def main() -> int:
    print(f"device {jax.devices()[0]} | mul {'clmad' if MUL is not field.mul else 'software'}")
    m, results = run(MUL)
    allok = True
    for nm, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}"); allok = allok and ok
    print(f"keccak stages A-B (commit + real-witness zerocheck) vs flock (m={m}): "
          f"{'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
