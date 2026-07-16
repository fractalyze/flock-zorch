"""GPU keccak3 LIGERITO prover timing (THE headline keccak path) vs flock keccak3
prove on the same instance. Byte-identity pinned by keccak3_ligerito_oracle_test
(walker probes + full R1csProofLigerito @m=22); this times the same code path at
the golden's size. Witness gen is ingested (not timed) on both sides.
Run: ... e2e_keccak3_ligerito_bench.py <flock_cpu_ms>"""
import sys
import numpy as np, frx
frx.config.update("jax_enable_x64", True)

import frx.numpy as jnp  # noqa: E402
from flock_zorch import zerocheck, lincheck, prover  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.lincheck.keccak3 import Keccak3LincheckCircuit  # noqa: E402
from flock_zorch.testing.keccak3_ligerito_oracle_test import load, _unpack  # noqa: E402
from flock_zorch.testing._util import best  # noqa: E402


def main():
    cpu = float(sys.argv[1]) if len(sys.argv) > 1 else None
    g = load(); meta = g["meta"]; cfg = g["cfg"]
    m = meta["m"]
    k_log, k_skip = meta["k_log"], meta["k_skip"]; ir = k_log - k_skip
    circ = Keccak3LincheckCircuit()
    a_bits, b_bits, c_bits = _unpack(g["a"], m), _unpack(g["b"], m), _unpack(g["z"], m)
    z, stmt = g["z"], g["stmt"]
    print(f"device {frx.devices()[0]} | keccak3-ligerito m={m}")

    def prove_once():
        root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, z)
        ch = Challenger(b"flock-keccak3-lig-v0"); prover.bind_statement(ch, stmt, root)
        zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch)
        x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
        _r, _zp, lcc, _zv = lincheck.prove(g["zlc"], None, None, x_ab, m, k_log, k_skip, ch=ch, capture=True, circuit=circ)
        ab = jnp.concatenate([lcc.r_inner_rest, x_ab.x_outer], axis=0)
        cc = jnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
        return prover.open_batch_ligerito(cfg, z, pdata, [ab, cc], ch)

    t = best(prove_once, n=3)
    sp = f"{cpu / t:.1f}x vs same-instance flock keccak3 Ligerito CPU {cpu:.0f}ms" if cpu else ""
    print(f"  GPU keccak3-Ligerito prove {t:8.2f} ms   {sp}")


if __name__ == "__main__":
    main()
