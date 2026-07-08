"""GPU SHA-256 LIGERITO prover timing (the HEADLINE path) vs flock prove_ligerito
on the same sha2 instance. Byte-identity pinned by sha2_ligerito_oracle_test.
Run: ... e2e_sha2_ligerito_bench.py <flock_cpu_ms>"""
import sys
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from flock_zorch import zerocheck, lincheck, prover  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.testing.sha2_ligerito_oracle_test import load, _unpack  # noqa: E402
from flock_zorch.testing._util import best  # noqa: E402

def main():
    cpu = float(sys.argv[1]) if len(sys.argv) > 1 else None
    g = load(); meta = g["meta"]; cfg = g["cfg"]
    m = meta["m"]; k_log, k_skip = meta["k_log"], meta["k_skip"]; ir = k_log - k_skip
    csc = lincheck.CscCircuit(g["a0_rows"], g["b0_rows"], 1 << k_log, const_pin=meta["const_pin"])
    a_bits, b_bits, c_bits = _unpack(g["a"], m), _unpack(g["b"], m), _unpack(g["z"], m)
    z, stmt = g["z"], g["stmt"]
    print(f"device {jax.devices()[0]} | sha2-ligerito m={m}")
    def prove_once():
        root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, z)
        ch = Challenger(b"flock-sha2-lig-v0"); prover.bind_statement(ch, stmt, root)
        zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch)
        x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
        _r, _zp, lcc, _zv = lincheck.prove(g["zlc"], None, None, x_ab, m, k_log, k_skip, ch=ch, capture=True, circuit=csc)
        ab = np.concatenate([lcc.r_inner_rest, x_ab.x_outer], axis=0)
        cc = np.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
        return prover.open_batch_ligerito(cfg, z, pdata, [ab, cc], ch)
    t = best(prove_once, n=3)
    sp = f"{cpu/t:.1f}x vs same-instance flock Ligerito CPU {cpu:.0f}ms" if cpu else ""
    print(f"  GPU sha256-Ligerito prove {t:8.2f} ms   {sp}")

if __name__ == "__main__":
    main()
