"""GPU SHA-256 prover with the LIGERITO PCS, byte gate vs flock prove_ligerito —
the HEADLINE sha256 path (M7).

Ingests dump_sha2_ligerito (real sha2 R1CS + Ligerito config + full
R1csProofLigerito), replays flock-zorch's prover on one shared challenger
(commit → bind → zerocheck → CSC lincheck → batched Ligerito open) and
byte-compares every field of the R1csProofLigerito.

Run (regen: cargo run --release --example dump_sha2_ligerito -- 128 artifacts/sha2_ligerito_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  FRX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/sha2_ligerito_oracle_test.py
"""
import sys

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402

from flock_zorch import ghash  # noqa: E402
from flock_zorch import zerocheck, lincheck, prover  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.testing._util import report  # noqa: E402
from flock_zorch.testing._golden import (  # noqa: E402
    ligerito_proof_results, open_golden, read_ligerito_config,
    read_ligerito_proof, unpack_bits)





def load(golden: str = "sha2_ligerito_golden.bin"):
    """Ingest a golden. `golden` names a file under `artifacts/`, so the same
    loader serves the m-variant dumps a size sweep needs."""
    rd = open_golden(golden)
    assert bytes(rd.take(8)) == b"FLKSL_01", "bad magic"
    meta = dict(m=rd.u(), k_log=rd.u(), k_skip=rd.u(), useful_bits=rd.u(), const_pin=rd.u(),
                lir=rd.u(), lbs=rd.u(), n_blocks_log=rd.u(), log_n=rd.u())
    cfg = read_ligerito_config(rd)
    g = dict(meta=meta, cfg=cfg, stmt=bytes(rd.raw(32)), root=rd.raw(32),
             z=rd.fv(), a=rd.fv(), b=rd.fv())
    g["zlc"] = bytes(rd.raw(rd.u()))
    g["a0_rows"] = rd.rowsu(); g["b0_rows"] = rd.rowsu()
    g["zc"] = dict(r1ab=rd.fv(), r1c=rd.fv(), mlv=rd.pair(), fa=rd.f(), fb=rd.f(), fc=rd.f())
    g["lc"] = dict(rounds=rd.pair(), zp=rd.fv())
    g["rs"] = [rd.fv() for _ in range(rd.u())]
    lig = read_ligerito_proof(rd)
    g["lig"] = lig
    return g


def run():
    g = load(); meta = g["meta"]; cfg = g["cfg"]
    m = meta["m"]
    k_log, k_skip = meta["k_log"], meta["k_skip"]; ir = k_log - k_skip
    results = []

    root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, g["z"])
    results.append(("commit root", np.array_equal(root, g["root"])))

    ch = Challenger(b"flock-sha2-lig-v0")
    prover.bind_statement(ch, g["stmt"], root)
    a_bits, b_bits, c_bits = g["a"], g["b"], g["z"]  # packed F128 — witness_to_rows unpacks on device
    zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch)
    results.append(("zerocheck round1_ab", np.array_equal(zc.round1_ab, g["zc"]["r1ab"])))
    results.append(("zerocheck final_c", np.array_equal(zc.final_c_eval, g["zc"]["fc"])))

    csc = lincheck.CscCircuit(g["a0_rows"], g["b0_rows"], 1 << k_log, const_pin=meta["const_pin"])
    x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
    _lr, lc_zp, lc_claim = lincheck.prove(g["zlc"], None, None, x_ab, m, k_log, k_skip, ch=ch, capture=True, circuit=csc)
    results.append(("lincheck z_partial", np.array_equal(ghash.to_lanes(lc_zp), g["lc"]["zp"])))

    ab_full = fnp.concatenate([lc_claim.r_inner_rest, x_ab.x_outer], axis=0)
    c_full = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
    out = prover.open_batch_ligerito(cfg, g["z"], pdata, [ab_full, c_full], ch)

    for i in range(len(g["rs"])):
        results.append((f"open ring_switch[{i}]", np.array_equal(ghash.to_lanes(out.ring_switches[i]), g["rs"][i])))
    p, gl = out.ligerito, g["lig"]

    results.extend(ligerito_proof_results(p, gl))
    return m, results


def main() -> int:
    print(f"device {frx.devices()[0]}")
    m, results = run()
    return report(results, f"sha2 LIGERITO full prove (R1csProofLigerito) vs flock prove_ligerito (m={m})")


if __name__ == "__main__":
    sys.exit(main())
