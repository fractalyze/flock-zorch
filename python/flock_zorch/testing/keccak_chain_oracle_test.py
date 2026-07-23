"""GPU keccak hash-CHAIN prover (Ligerito) byte gate vs flock KeccakSetup::
prove_chain (task #14, M4b). Proves 2^n keccaks form a sequential chain
x_{i+1}=keccak_f(x_i) with public endpoints.

Replays the full chain prover on one shared challenger:
  commit → bind → zerocheck → walker lincheck → τ_pos → region fold → shift
  sumcheck → MIXED open (ab,c ring-switched + chain packed-direct)
and byte-compares the full ChainProofLigerito {zerocheck, lincheck, shift,
pcs_open(ligerito)}.

Run (regen: cargo run --release --example dump_keccak_chain -- artifacts/keccak_chain_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  FRX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/keccak_chain_oracle_test.py
"""
import sys

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402

from flock_zorch import zerocheck, lincheck, prover, ghash  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.lincheck import chain  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.testing._util import report  # noqa: E402
from flock_zorch.testing._golden import (  # noqa: E402
    ligerito_proof_results, open_golden, read_ligerito_config,
    read_ligerito_proof, unpack_bits)
from flock_zorch.lincheck.keccak import KeccakLincheckCircuit  # noqa: E402





def load():
    rd = open_golden("keccak_chain_golden.bin")
    assert bytes(rd.take(8)) == b"FLKKC_01", "bad magic"
    meta = dict(m=rd.u(), k_log=rd.u(), k_skip=rd.u(), useful_bits=rd.u(), const_pin=rd.u(),
                lir=rd.u(), lbs=rd.u(), n_blocks_log=rd.u(), log_n=rd.u(),
                region_log=rd.u(), input_byte_off=rd.u(), output_byte_off=rd.u())
    cfg = read_ligerito_config(rd)
    g = dict(meta=meta, cfg=cfg, stmt=bytes(rd.raw(32)), root=rd.raw(32),
             z=rd.fv(), a=rd.fv(), b=rd.fv())
    g["zlc"] = bytes(rd.raw(rd.u()))
    g["zc"] = dict(r1ab=rd.fv(), r1c=rd.fv(), mlv=rd.pair(), fa=rd.f(), fb=rd.f(), fc=rd.f())
    g["lc"] = dict(rounds=rd.pair(), zp=rd.fv())
    g["shift"] = dict(rounds=rd.pair(), g_at_point=rd.f())
    g["rs"] = [rd.fv() for _ in range(rd.u())]
    lig = read_ligerito_proof(rd)
    g["lig"] = lig
    return g


def run():
    g = load(); meta = g["meta"]; cfg = g["cfg"]
    m = meta["m"]
    k_log, k_skip = meta["k_log"], meta["k_skip"]; ir = k_log - k_skip   # inner_rest = 10
    region_log = meta["region_log"]
    results = []

    root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, g["z"])
    results.append(("commit root", np.array_equal(root, g["root"])))

    ch = Challenger(b"flock-keccak-chain-v0")
    prover.bind_statement(ch, g["stmt"], root)
    a_bits, b_bits, c_bits = unpack_bits(g["a"], m), unpack_bits(g["b"], m), unpack_bits(g["z"], m)
    zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch)
    results.append(("zerocheck round1_ab", np.array_equal(ghash.to_lanes(zc.round1_ab), g["zc"]["r1ab"])))
    results.append(("zerocheck final_c", np.array_equal(ghash.to_lanes(zc.final_c_eval).reshape(2), g["zc"]["fc"])))

    circ = KeccakLincheckCircuit()
    x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
    _lr, lc_zp, lc_claim = lincheck.prove(g["zlc"], None, None, x_ab, m, k_log, k_skip, ch=ch, circuit=circ)
    results.append(("lincheck z_partial", np.array_equal(ghash.to_lanes(lc_zp), g["lc"]["zp"])))

    # ---- chain: τ_pos → region fold → shift sumcheck → assemble packed-direct claim
    tau_pos = ch.sample_f128(region_log - chain.LOG_PACKING)
    in_vals, out_vals = chain.fold_in_out(g["z"], k_log, tau_pos,
                                          meta["input_byte_off"], meta["output_byte_off"])
    sh_rounds, _g_at, sh_claims = chain.prove_chain_shift(in_vals, out_vals, ch)
    got_sr = np.array([np.concatenate([ghash.to_lanes(e1).reshape(2), ghash.to_lanes(ei).reshape(2)])
                       for e1, ei in sh_rounds]) if sh_rounds else np.zeros((0, 4), np.uint64)
    want_sr = np.array([np.concatenate([e1, ei]) for e1, ei in g["shift"]["rounds"]]) if g["shift"]["rounds"] else np.zeros((0, 4), np.uint64)
    results.append(("shift rounds", got_sr.shape == want_sr.shape and np.array_equal(got_sr, want_sr)))
    results.append(("shift g_at_point", np.array_equal(ghash.to_lanes(sh_claims["value"]).reshape(2), g["shift"]["g_at_point"])))
    chain_claim = chain.assemble_chain_claim(tau_pos, sh_claims, k_log, region_log)

    # ---- mixed open: [ab, c] ring-switched + [chain] packed-direct
    ab_full = fnp.concatenate([lc_claim.r_inner_rest, x_ab.x_outer], axis=0)
    c_full = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
    out = prover.open_batch_mixed_ligerito(cfg, g["z"], pdata,
                                           [ab_full, c_full], [chain_claim], ch)

    for i in range(len(g["rs"])):
        results.append((f"open ring_switch[{i}]", np.array_equal(ghash.to_lanes(out.ring_switches[i]), g["rs"][i])))
    p, gl = out.ligerito, g["lig"]

    results.extend(ligerito_proof_results(p, gl))
    return m, results


def main() -> int:
    print(f"device {frx.devices()[0]}")
    m, results = run()
    return report(results, f"keccak CHAIN prove (ChainProofLigerito: zc+lc+shift+mixed open) "
                           f"vs flock prove_chain (m={m})")


if __name__ == "__main__":
    sys.exit(main())
