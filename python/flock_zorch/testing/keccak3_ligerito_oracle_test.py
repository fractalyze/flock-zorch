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

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402

from flock_zorch import zerocheck, lincheck, prover, ghash  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.testing._util import report  # noqa: E402
from flock_zorch.testing._golden import (  # noqa: E402
    ligerito_proof_results, open_golden, read_ligerito_config,
    read_ligerito_proof)
from flock_zorch.lincheck.keccak3 import Keccak3LincheckCircuit  # noqa: E402





def load(golden: str = "keccak3_ligerito_golden.bin"):
    """Ingest a golden. `golden` names a file under `artifacts/`, so the same
    loader serves the m-variant dumps a size sweep needs."""
    rd = open_golden(golden)
    assert bytes(rd.take(8)) == b"FLKK3L01", "bad magic"
    meta = dict(m=rd.u(), k_log=rd.u(), k_skip=rd.u(), useful_bits=rd.u(), const_pin=rd.u(),
                lir=rd.u(), lbs=rd.u(), n_blocks_log=rd.u(), log_n=rd.u())
    cfg = read_ligerito_config(rd)
    g = dict(meta=meta, cfg=cfg, stmt=bytes(rd.raw(32)), root=rd.raw(32),
             z=rd.fv(), a=rd.fv(), b=rd.fv())
    g["zlc"] = bytes(rd.raw(rd.u()))
    g["probes"] = [dict(alpha=rd.f(), eq=rd.fv(), comb=rd.fv()) for _ in range(rd.u())]
    g["zc"] = dict(r1ab=rd.fv(), r1c=rd.fv(), mlv=rd.pair(), fa=rd.f(), fb=rd.f(), fc=rd.f())
    g["lc"] = dict(rounds=rd.pair(), zp=rd.fv())
    g["rs"] = [rd.fv() for _ in range(rd.u())]
    lig = read_ligerito_proof(rd)
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
    a_bits, b_bits, c_bits = g["a"], g["b"], g["z"]
    zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch)
    results.append(("zerocheck round1_ab", np.array_equal(zc.round1_ab, g["zc"]["r1ab"])))
    results.append(("zerocheck final_c", np.array_equal(zc.final_c_eval, g["zc"]["fc"])))

    circ = Keccak3LincheckCircuit()
    x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
    _lr, lc_zp, lc_claim = lincheck.prove(g["zlc"], None, None, x_ab, m, k_log, k_skip, ch=ch, circuit=circ)
    results.append(("lincheck z_partial", np.array_equal(ghash.to_lanes(lc_zp), g["lc"]["zp"])))

    ab_full = fnp.concatenate([lc_claim.r_inner_rest, x_ab.x_outer], axis=0)
    c_full = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
    out = prover.open_batch_ligerito(cfg, g["z"], pdata, [ab_full, c_full], ch)

    for i in range(len(g["rs"])):
        results.append((f"open ring_switch[{i}]", np.array_equal(ghash.to_lanes(out.ring_switches[i]), g["rs"][i])))
    p, gl = out.ligerito, g["lig"]

    results.extend(ligerito_proof_results(p, gl))

    # Stage W: the keccak3 walker port — Keccak3LincheckCircuit.fold_alpha_batched (standalone)
    for i, pb in enumerate(g["probes"]):
        comb = ghash.from_ghash_host(circ.fold_alpha_batched(ghash.to_ghash(pb["alpha"]), ghash.to_ghash(pb["eq"])))
        results.append((f"walker probe {i} (fold_alpha_batched)", np.array_equal(comb, pb["comb"])))
    return m, results


def main() -> int:
    print(f"device {frx.devices()[0]}")
    m, results = run()
    return report(results, f"keccak3 LIGERITO full prove (R1csProofLigerito) vs flock prove_fast (m={m})")


if __name__ == "__main__":
    sys.exit(main())
