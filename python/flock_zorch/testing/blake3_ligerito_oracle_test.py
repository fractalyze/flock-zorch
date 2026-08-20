"""GPU BLAKE3 prover with the LIGERITO PCS, byte gate vs flock prove_ligerito —
the HEADLINE blake3 path, mirroring sha2_ligerito_oracle_test.py.

Ingests dump_blake3_ligerito (real blake3 R1CS + Ligerito config + full
R1csProofLigerito), replays flock-zorch's prover on one shared challenger
(commit → bind → zerocheck → CSC lincheck → batched Ligerito open) and
byte-compares every field of the R1csProofLigerito. BLAKE3 mirrors sha2: its
a_0/b_0 are populated sparse matrices folded by the generic `CscCircuit`.

Run (regen: cargo run --release --example dump_blake3_ligerito -- 256 \
artifacts/blake3_ligerito_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  FRX_PLATFORMS=cuda,cpu PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/blake3_ligerito_oracle_test.py
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
    zerocheck,
)
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.r1cs_hashes import blake3_witness  # noqa: E402
from flock_zorch.sha256_challenger import Sha256Challenger  # noqa: E402
from flock_zorch.testing._golden import (  # noqa: E402
    latest_blake3_golden,
    ligerito_proof_results,
)

# The loader moved to `_golden` so a Bazel target can depend on it; this keeps
# `load` importable from here, which is the name the gates already use.
load = latest_blake3_golden
from flock_zorch.testing._util import gate_device, report  # noqa: E402


def substitute_device_witness(g):
    """Regenerate the witness on device from the golden's own blocks and swap
    it into `g`, so the standard gates below exercise the witgen path.

    The compression inputs come out of the golden's z prefix
    (`blake3_witness.extract_inputs`), so no extra fixture exists to go stale. The
    proof gates then transitively pin witgen against flock: one diverging
    witness bit flips every Fiat-Shamir draw after it. This is what catches
    witgen drifting from flock's layout across a pin bump + golden re-dump —
    an event every golden-fed gate is green through by construction.
    """
    z, a, b = blake3_witness.witness_blake3(*blake3_witness.extract_inputs(g["z"]))
    # Dispatch the stripe before the blocking D2H compare pulls below, so it
    # overlaps them instead of serializing after ~1.5 GiB of transfer.
    zlc = blake3_witness.lincheck_stripe(z)
    z, a, b = (x.reshape(-1, 2) for x in (z, a, b))
    checks = [
        (f"witgen {k} vs golden", np.array_equal(np.asarray(v), g[k]))
        for k, v in zip("zab", (z, a, b))
    ]
    ref = np.frombuffer(g["zlc"], np.uint8).reshape(
        -1, blake3_witness.STRIPE_BYTES_PER_GROUP
    )
    checks.append(("witgen zlc vs golden", np.array_equal(np.asarray(zlc), ref)))
    g["z"], g["a"], g["b"] = z, a, b
    g["zlc"] = zlc
    return checks


def run(golden: str = "blake3_ligerito_golden.bin", device_witness: bool = False):
    g = load(golden)
    meta = g["meta"]
    cfg = g["cfg"]
    m = meta["m"]
    k_log, k_skip = meta["k_log"], meta["k_skip"]
    ir = k_log - k_skip
    results = substitute_device_witness(g) if device_witness else []

    root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, g["z"])
    results.append(("commit root", np.array_equal(root, g["root"])))

    ch = Sha256Challenger(b"flock-blake3-lig-v0")
    prover.bind_statement(ch, g["stmt"], root)
    a_bits, b_bits, c_bits = (
        g["a"],
        g["b"],
        g["z"],
    )  # packed F128 — witness_to_rows unpacks on device
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

    csc = lincheck.CscCircuit(
        g["a0_rows"], g["b0_rows"], 1 << k_log, const_pin=meta["const_pin"]
    )
    x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
    lc = lincheck.prove(
        g["zlc"], None, None, x_ab, m, k_log, k_skip, ch=ch, circuit=csc
    )
    assert lc.claim is not None, "full lincheck prove always yields a claim"
    results.append(
        (
            "lincheck z_partial",
            np.array_equal(ghash.to_lanes(lc.z_partial), g["lc"]["zp"]),
        )
    )

    ab_full = fnp.concatenate([lc.claim.r_inner_rest, x_ab.x_outer], axis=0)
    c_full = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
    # Thread the lincheck z_vec exactly as `prove_fast` does, so this gate
    # byte-checks the precomputed-s_hat_v open against the flock golden.
    out = prover.open_batch_ligerito(
        cfg,
        g["z"],
        pdata,
        [ab_full, c_full],
        ch,
        precomputed_s_hat_vs=prover.ab_precomputed_s_hat_vs(
            lc.z_vec, lc.claim.r_inner_rest
        ),
    )

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
        "--golden",
        default="blake3_ligerito_golden.bin",
        help="golden filename under artifacts/ (the m-variant dumps)",
    )
    ap.add_argument(
        "--witgen",
        action="store_true",
        help="regenerate the witness on device from the golden's own blocks "
        "(gates flock_zorch.r1cs_hashes.blake3_witness against flock end to end)",
    )
    args = ap.parse_args()
    gate_device()
    m, results = run(args.golden, device_witness=args.witgen)
    return report(
        results,
        f"blake3 LIGERITO full prove (R1csProofLigerito) vs flock "
        f"prove_ligerito (m={m})",
    )


if __name__ == "__main__":
    sys.exit(main())
