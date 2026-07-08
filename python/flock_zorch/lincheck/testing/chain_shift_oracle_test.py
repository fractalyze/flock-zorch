"""GPU hash-chain prover-core byte gate vs flock (task #14, M4a): the shift
sumcheck (chain.prove_chain_shift) and the region fold (chain.fold_in_out) — the
two new prover pieces of the keccak hash-CHAIN protocol.
  Gate A: prove_chain_shift on dumped In/Out → ChainShiftProof rounds/g_at_point
          + ChainClaims (instance_point, sel0, value), on a shared challenger.
  Gate B: fold_in_out on a dumped packed witness + τ_pos → (in_vals, out_vals).

Run (regen: cargo run --release --example dump_chain_shift -- artifacts/chain_shift_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/lincheck/testing/chain_shift_oracle_test.py
"""
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from flock_zorch import field  # noqa: E402
from flock_zorch.lincheck import chain  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402

ART = Path(__file__).resolve().parents[4] / "artifacts"


class R:
    def __init__(self, buf): self.b = buf; self.o = 0
    def take(self, n): v = self.b[self.o:self.o + n]; self.o += n; return v
    def u(self): return int.from_bytes(self.take(8), "little")
    def f(self): return np.frombuffer(self.take(16), np.uint64).copy()
    def fv(self): n = self.u(); return np.frombuffer(self.take(16 * n), np.uint64).reshape(n, 2).copy()
    def pair(self): n = self.u(); return [(self.f(), self.f()) for _ in range(n)]


def load():
    rd = R((ART / "chain_shift_golden.bin").read_bytes())
    assert bytes(rd.take(8)) == b"FLKCHN01", "bad magic"
    a = dict(n=rd.u(), in_vals=rd.fv(), out_vals=rd.fv(),
             rounds=rd.pair(), g_at_point=rd.f(),
             instance_point=rd.fv(), sel0=rd.f(), value=rd.f())
    b = dict(k_log=rd.u(), region_log=rd.u(), input_byte_off=rd.u(), output_byte_off=rd.u(),
             tau_pos=rd.fv(), packed=rd.fv(), in_vals=rd.fv(), out_vals=rd.fv())
    return a, b


def run():
    ga, gb = load()
    results = []

    # ---- Gate A: shift sumcheck on a fresh shared challenger.
    ch = Challenger(b"flock-chain-shift-v0")
    rounds, g_at, claims = chain.prove_chain_shift(ga["in_vals"], ga["out_vals"], ch)
    got_r = np.array([np.concatenate([e1, ei]) for e1, ei in rounds]) if rounds else np.zeros((0, 4), np.uint64)
    want_r = np.array([np.concatenate([e1, ei]) for e1, ei in ga["rounds"]]) if ga["rounds"] else np.zeros((0, 4), np.uint64)
    results.append(("shift rounds", got_r.shape == want_r.shape and np.array_equal(got_r, want_r)))
    results.append(("shift g_at_point", np.array_equal(g_at, ga["g_at_point"])))
    results.append(("shift claim.instance_point", np.array_equal(claims["instance_point"], ga["instance_point"])))
    results.append(("shift claim.sel0", np.array_equal(claims["sel0"], ga["sel0"])))
    results.append(("shift claim.value", np.array_equal(claims["value"], ga["value"])))

    # ---- Gate B: region fold.
    fin, fout = chain.fold_in_out(gb["packed"], gb["k_log"], gb["tau_pos"],
                                  gb["input_byte_off"], gb["output_byte_off"])
    results.append(("fold in_vals", np.array_equal(fin, gb["in_vals"])))
    results.append(("fold out_vals", np.array_equal(fout, gb["out_vals"])))
    return ga["n"], results


def main() -> int:
    print(f"device {jax.devices()[0]}")
    n, results = run()
    allok = True
    for nm, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}"); allok = allok and ok
    print(f"chain shift core (prove_chain_shift + fold_in_out) vs flock (n={n}): "
          f"{'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
