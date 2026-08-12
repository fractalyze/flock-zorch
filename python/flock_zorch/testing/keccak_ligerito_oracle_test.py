"""GPU Keccak-f[1600] prover with the LIGERITO PCS, byte gate vs flock
KeccakSetup::prove_fast (Ligerito) — the headline keccak backend. Task #14, M3a.

Ingests dump_keccak_ligerito (keccak R1CS + Ligerito config + full
R1csProofLigerito), replays flock-zorch's prover on one shared challenger
(commit → bind → zerocheck → walker lincheck → batched Ligerito open) and
byte-compares every field of the R1csProofLigerito. Keccak's A_0/B_0 are empty
stubs, so the lincheck circuit is the procedural KeccakLincheckCircuit (no a0/b0
rows); everything else is the same recursive Ligerito driver gated for sha2.

Run (regen: cargo run --release --example dump_keccak_ligerito -- 64 \
artifacts/keccak_ligerito_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  FRX_PLATFORMS=cuda,cpu PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/keccak_ligerito_oracle_test.py

`--witgen` additionally regenerates the witness on device from the golden's own
states, which puts `flock_zorch.witgen_keccak` under the same proof-level gate.
"""

import argparse
import sys

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402

from flock_zorch import (  # noqa: E402
    ghash,  # noqa: E402
    lincheck,
    prover,
    witgen_keccak,
    zerocheck,
)
from flock_zorch.lincheck.keccak import KeccakLincheckCircuit  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.sha256_challenger import Sha256Challenger  # noqa: E402
from flock_zorch.testing._golden import (  # noqa: E402
    ligerito_proof_results,
    open_golden,
    read_ligerito_config,
    read_ligerito_proof,
    unpack_bits,
)
from flock_zorch.testing._util import report  # noqa: E402


def load():
    rd = open_golden("keccak_ligerito_golden.bin")
    assert bytes(rd.take(8)) == b"FLKKL_01", "bad magic"
    meta = dict(
        m=rd.u(),
        k_log=rd.u(),
        k_skip=rd.u(),
        useful_bits=rd.u(),
        const_pin=rd.u(),
        lir=rd.u(),
        lbs=rd.u(),
        n_blocks_log=rd.u(),
        log_n=rd.u(),
    )
    cfg = read_ligerito_config(rd)
    g = dict(
        meta=meta,
        cfg=cfg,
        stmt=bytes(rd.raw(32)),
        root=rd.raw(32),
        z=rd.fv(),
        a=rd.fv(),
        b=rd.fv(),
    )
    g["zlc"] = bytes(rd.raw(rd.u()))
    g["zc"] = dict(
        r1ab=rd.fv(), r1c=rd.fv(), mlv=rd.pair(), fa=rd.f(), fb=rd.f(), fc=rd.f()
    )
    g["lc"] = dict(rounds=rd.pair(), zp=rd.fv())
    g["rs"] = [rd.fv() for _ in range(rd.u())]
    lig = read_ligerito_proof(rd)
    g["lig"] = lig
    return g


def substitute_device_witness(g):
    """Regenerate the witness on device from the golden's own states and swap it
    into `g`, so the gates below exercise the witgen path.

    `state_0` comes straight out of the golden's own z prefix — keccak writes
    whole lanes, so the first 25 are the input states verbatim and no extra
    fixture exists to go stale. The proof gates then transitively pin
    `witgen_keccak` against flock: one diverging witness bit flips every
    Fiat-Shamir draw after it, which is what catches the layout drifting across
    a pin bump plus golden re-dump.

    `zlc` is left as the golden's. The keccak lincheck stripe is a different
    shape from BLAKE3's and has no device port yet, so substituting it here
    would gate something that does not exist.
    """
    zw = np.asarray(g["z"], dtype=np.uint64).reshape(-1, witgen_keccak.WORDS_PER_BLOCK)
    state0 = zw[:, witgen_keccak.STATE0_LANE : witgen_keccak.STATE0_LANE + 25]
    z, a, b = (
        np.asarray(x).reshape(-1, 2)
        for x in witgen_keccak.witness_keccak(frx.device_put(state0))
    )
    checks = [
        (f"witgen_keccak {k} vs golden", np.array_equal(v, g[k]))
        for k, v in zip("zab", (z, a, b))
    ]
    g["z"], g["a"], g["b"] = z, a, b
    return checks


def run(device_witness=False):
    g = load()
    meta = g["meta"]
    cfg = g["cfg"]
    m = meta["m"]
    k_log, k_skip = meta["k_log"], meta["k_skip"]
    ir = k_log - k_skip  # inner_rest = 16 - 6 = 10
    results = substitute_device_witness(g) if device_witness else []

    root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, g["z"])
    results.append(("commit root", np.array_equal(root, g["root"])))

    ch = Sha256Challenger(b"flock-keccak-lig-v0")
    prover.bind_statement(ch, g["stmt"], root)
    a_bits, b_bits, c_bits = (
        unpack_bits(g["a"], m),
        unpack_bits(g["b"], m),
        unpack_bits(g["z"], m),
    )
    zc_proof, zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch)
    results.append(
        (
            "zerocheck round1_ab",
            np.array_equal(ghash.to_lanes(zc_proof.round1_ab), g["zc"]["r1ab"]),
        )
    )
    results.append(
        (
            "zerocheck final_c",
            np.array_equal(
                ghash.to_lanes(zc_proof.final_c_eval).reshape(2), g["zc"]["fc"]
            ),
        )
    )

    circ = KeccakLincheckCircuit()
    x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
    _lr, lc_zp, lc_claim = lincheck.prove(
        g["zlc"], None, None, x_ab, m, k_log, k_skip, ch=ch, circuit=circ
    )
    results.append(
        ("lincheck z_partial", np.array_equal(ghash.to_lanes(lc_zp), g["lc"]["zp"]))
    )

    ab_full = fnp.concatenate([lc_claim.r_inner_rest, x_ab.x_outer], axis=0)
    c_full = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
    out = prover.open_batch_ligerito(cfg, g["z"], pdata, [ab_full, c_full], ch)

    for i in range(len(g["rs"])):
        results.append(
            (
                f"open ring_switch[{i}]",
                np.array_equal(ghash.to_lanes(out.ring_switches[i]), g["rs"][i]),
            )
        )
    p, gl = out.ligerito, g["lig"]

    results.extend(ligerito_proof_results(p, gl))
    return m, results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--witgen",
        action="store_true",
        help="regenerate the witness on device from the golden's own states "
        "(gates flock_zorch.witgen_keccak against flock end to end)",
    )
    args = ap.parse_args()
    print(f"device {frx.devices()[0]}")
    m, results = run(device_witness=args.witgen)
    return report(
        results,
        f"keccak LIGERITO full prove (R1csProofLigerito) vs flock prove_fast (m={m})",
    )


if __name__ == "__main__":
    sys.exit(main())
