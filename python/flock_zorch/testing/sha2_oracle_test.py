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
  JAX_PLATFORMS=cuda PYTHONPATH=python:/home/jooman/fractalyze/zorch <venv> \
      python/flock_zorch/testing/sha2_oracle_test.py
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
    return rd, g


def run(mul):
    _, g = load()
    m = g["meta"]["m"]; lir = g["meta"]["lir"]; lbs = g["meta"]["lbs"]
    results = []

    # Stage A: commit root on the real packed witness
    root, codeword, tree = pcs_commit.commit(g["z"], m, lir, lbs, mul=mul)
    _eq("commit root", root, g["root"], results)

    # Stage B: zerocheck on real a=A·z, b=B·z, c=z (useful_bits padding) — shared challenger
    ch = Challenger(b"flock-sha2-v0")
    prover.bind_statement(ch, g["stmt"], root)
    a_bits = _unpack(g["a"], m); b_bits = _unpack(g["b"], m); c_bits = _unpack(g["z"], m)
    zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, mul=mul, ch=ch)
    _eq("zc round1_ab", zc["round1_ab"], g["zc"]["r1ab"], results)
    _eq("zc round1_c", zc["round1_c"], g["zc"]["r1c"], results)
    got_mlv = np.array([np.concatenate([a, b]) for a, b in zc["multilinear_rounds"]])
    want_mlv = np.array([np.concatenate([a, b]) for a, b in g["zc"]["mlv"]])
    _eq("zc multilinear_rounds", got_mlv, want_mlv, results)
    _eq("zc final_a", zc["final_a_eval"], g["zc"]["fa"], results)
    _eq("zc final_b", zc["final_b_eval"], g["zc"]["fb"], results)
    _eq("zc final_c", zc["final_c_eval"], g["zc"]["fc"], results)
    return m, results


def main() -> int:
    print(f"device {jax.devices()[0]} | mul {'clmad' if MUL is not field.mul else 'software'}")
    m, results = run(MUL)
    allok = True
    for nm, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}"); allok = allok and ok
    print(f"sha2 stages A-B (commit + real-witness zerocheck) vs flock (m={m}): "
          f"{'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
