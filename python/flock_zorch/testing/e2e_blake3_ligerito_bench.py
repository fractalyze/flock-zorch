"""GPU BLAKE3 LIGERITO prover timing — mirrors e2e_sha2_ligerito_bench with the
blake3_ligerito loader (byte-identity pinned by blake3_ligerito_oracle_test)."""
import sys
import numpy as np, frx
frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
from flock_zorch import zerocheck, lincheck, prover  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.challenger import flock_transcript  # noqa: E402
from flock_zorch.testing.blake3_ligerito_oracle_test import load  # noqa: E402
from flock_zorch.testing._util import best  # noqa: E402


def main():
    cpu = float(sys.argv[1]) if len(sys.argv) > 1 else None
    g = load(); meta = g["meta"]; cfg = g["cfg"]
    m = meta["m"]; k_log, k_skip = meta["k_log"], meta["k_skip"]; ir = k_log - k_skip
    csc = lincheck.CscCircuit(g["a0_rows"], g["b0_rows"], 1 << k_log, const_pin=meta["const_pin"])
    a_bits, b_bits, c_bits = g["a"], g["b"], g["z"]  # packed F128 — witness_to_rows unpacks on device
    z, stmt = g["z"], g["stmt"]
    print(f"device {frx.devices()[0]} | blake3-ligerito m={m}")

    def prove_once():
        root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, z)
        transcript = flock_transcript(b"flock-blake3-lig-v0")
        transcript = prover.bind_statement(transcript, stmt, root)
        zc, transcript = zerocheck.prove_packed(
            a_bits, b_bits, c_bits, m, transcript=transcript
        )
        x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
        lp, transcript = lincheck.prove(g["zlc"], None, None, x_ab, m, k_log, k_skip, transcript=transcript, capture=True, circuit=csc)
        _r, _zp, lcc, _zv = lp
        ab = fnp.concatenate([lcc.r_inner_rest, x_ab.x_outer], axis=0)
        cc = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
        proof, _transcript = prover.open_batch_ligerito(
            cfg, z, pdata, [ab, cc], transcript
        )
        return proof

    t = best(prove_once, n=3)
    sp = f"{cpu/t:.1f}x vs same-instance flock Ligerito CPU {cpu:.0f}ms" if cpu else ""
    print(f"  GPU blake3-Ligerito prove {t:8.2f} ms   {sp}")


if __name__ == "__main__":
    main()
