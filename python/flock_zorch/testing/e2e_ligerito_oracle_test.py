"""GPU fused R1CS prover (`prover.prove_fast`) on the LIGERITO PCS, byte gate vs
flock `prover::prove_fast_ligerito` on the identity R1CS — the e2e gate after the
BaseFold backend was removed.

Ingests dump_e2e_ligerito (identity R1CS + Ligerito config + full
R1csProofLigerito), drives flock-zorch's fused `prove_fast` (Ligerito commit →
bind → zerocheck → dense lincheck → batched Ligerito open) on one challenger and
byte-compares every proof field. Identity R1CS: A_0=B_0=C_0=I (a=b=c=z), dense
a0/b0=I, no lincheck circuit.

Run (regen: cargo run --release --example dump_e2e_ligerito -- 22 artifacts/e2e_ligerito_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  FRX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/e2e_ligerito_oracle_test.py
"""
import sys

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch import prover, ghash  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.testing._golden import (  # noqa: E402
    ligerito_proof_results, open_golden, read_ligerito_config,
    read_ligerito_proof, unpack_bits)
from flock_zorch.testing._util import report  # noqa: E402

DOMAIN = b"flock-e2e-lig-v0"


def load():
    rd = open_golden("e2e_ligerito_golden.bin")
    assert bytes(rd.take(8)) == b"FLKE2L01", "bad magic"
    meta = dict(m=rd.u(), k_log=rd.u(), k_skip=rd.u(), useful_bits=rd.u(),
                lir=rd.u(), lbs=rd.u(), n_blocks_log=rd.u(), log_n=rd.u())
    cfg = read_ligerito_config(rd)
    g = dict(meta=meta, cfg=cfg, stmt=bytes(rd.raw(32)), root=rd.raw(32),
             z=rd.fv(), a=rd.fv(), b=rd.fv())
    g["zlc"] = bytes(rd.raw(rd.u()))
    g["zc"] = dict(r1ab=rd.fv(), r1c=rd.fv(), mlv=rd.pair(), fa=rd.f(), fb=rd.f(), fc=rd.f())
    g["lc"] = dict(rounds=rd.pair(), zp=rd.fv())
    g["ab_v"] = rd.f(); g["c_v"] = rd.f()
    g["rs"] = [rd.fv() for _ in range(rd.u())]
    lig = read_ligerito_proof(rd)
    g["lig"] = lig
    return g


def _lanes(x):
    """Native ghash F128(s) -> host uint64 [-1, 2]."""
    a = ghash.to_lanes(x)
    return np.asarray(a).reshape(-1, 2)


def run():
    g = load(); meta = g["meta"]; cfg = g["cfg"]
    m, k_log, k_skip, ub = meta["m"], meta["k_log"], meta["k_skip"], meta["useful_bits"]
    results = []

    root, _pdata = zorch_ligerito.commit_flock_ligerito(cfg, g["z"])
    results.append(("commit root", np.array_equal(root, g["root"])))

    k = 1 << k_log
    a0 = np.eye(k, dtype=np.uint64)
    b0 = np.eye(k, dtype=np.uint64)
    res = prover.prove_fast(g["z"], m, k_log, k_skip, a0, b0, g["zlc"], g["stmt"],
                            cfg, domain=DOMAIN)

    zc, gzc = res.zerocheck, g["zc"]
    results.append(("zc round1_ab", np.array_equal(_lanes(zc.round1_ab), gzc["r1ab"])))
    results.append(("zc round1_c", np.array_equal(_lanes(zc.round1_c), gzc["r1c"])))
    got_mlv = np.array([np.concatenate([_lanes(a).reshape(2), _lanes(b).reshape(2)]) for a, b in zc.multilinear_rounds])
    want_mlv = np.array([np.concatenate([a, b]) for a, b in gzc["mlv"]])
    results.append(("zc multilinear_rounds", got_mlv.shape == want_mlv.shape and np.array_equal(got_mlv, want_mlv)))
    results.append(("zc final_a", np.array_equal(_lanes(zc.final_a_eval).reshape(2), gzc["fa"])))
    results.append(("zc final_b", np.array_equal(_lanes(zc.final_b_eval).reshape(2), gzc["fb"])))
    results.append(("zc final_c", np.array_equal(_lanes(zc.final_c_eval).reshape(2), gzc["fc"])))

    lc_rounds, lc_zp = res.lincheck
    got_lcr = np.array([np.concatenate([_lanes(a).reshape(2), _lanes(b).reshape(2)]) for a, b in lc_rounds]) \
        if lc_rounds else np.zeros((0, 4), np.uint64)
    want_lcr = np.array([np.concatenate([a, b]) for a, b in g["lc"]["rounds"]]) \
        if g["lc"]["rounds"] else np.zeros((0, 4), np.uint64)
    results.append(("lc rounds", got_lcr.shape == want_lcr.shape and np.array_equal(got_lcr, want_lcr)))
    results.append(("lc z_partial", np.array_equal(_lanes(lc_zp), g["lc"]["zp"])))
    results.append(("claim ab.value", np.array_equal(_lanes(res.claim_ab_value).reshape(2), g["ab_v"])))
    results.append(("claim c.value", np.array_equal(_lanes(res.claim_c_value).reshape(2), g["c_v"])))

    out = res.pcs_open
    for i in range(len(g["rs"])):
        results.append((f"open ring_switch[{i}]", np.array_equal(ghash.to_lanes(out.ring_switches[i]), g["rs"][i])))
    p, gl = out.ligerito, g["lig"]

    results.extend(ligerito_proof_results(p, gl))
    return m, results


def main() -> int:
    print(f"device {frx.devices()[0]}")
    m, results = run()
    return report(results, f"e2e LIGERITO fused prove (prove_fast) vs flock prove_fast_ligerito (identity m={m})")


if __name__ == "__main__":
    sys.exit(main())
