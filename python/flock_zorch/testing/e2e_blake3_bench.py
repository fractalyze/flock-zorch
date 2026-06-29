"""GPU BLAKE3 prover timing (commit→zerocheck→CSC lincheck→batched open) vs
flock's BaseFold `prove` on the SAME real blake3 R1CS. Mirrors e2e_sha2_bench.py.

Ingests the dump_blake3 golden (witness + a=A·z + matrices are host setup), then
times the GPU prover phases on one shared challenger. Byte-identity is pinned by
blake3_oracle_test. CPU baseline: bench_blake3_cpu.rs (same instance, BaseFold).

HONEST scope: the timed GPU number is the PROVER (commit/zerocheck/lincheck/open);
the host matvec a=A·z (flock witness-prep) is excluded. The CSC lincheck fold now
runs ON DEVICE (lincheck.CscCircuit sorted prefix-XOR scan, ~0.5 ms vs the old
491 ms host np.bitwise_xor.at over blake3's ~21M nonzeros).

Run:
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH=python:/home/jooman/fractalyze/zorch <venv> \
      python/flock_zorch/testing/e2e_blake3_bench.py [cpu_ms]
"""
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from flock_zorch import field, pcs_commit, zerocheck, lincheck, prover  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.testing.blake3_oracle_test import load, _unpack  # noqa: E402
from flock_zorch.testing._util import best, select_mul  # noqa: E402

ART = Path(__file__).resolve().parents[3] / "artifacts"

MUL = select_mul()


def main():
    cpu = float(sys.argv[1]) if len(sys.argv) > 1 else None
    _, g = load()
    meta = g["meta"]; m = meta["m"]; lir = meta["lir"]; lbs = meta["lbs"]
    k_log = meta["k_log"]; k_skip = meta["k_skip"]; ir = k_log - k_skip
    k_code = (m - 7 - lbs) + lir
    print(f"device {jax.devices()[0]} | mul {'clmad' if MUL is not field.mul else 'software'} | blake3 m={m}")

    # setup (host witness-prep / circuit build — not the GPU prover, not timed)
    csc = lincheck.CscCircuit(g["a0_rows"], g["b0_rows"], 1 << k_log, const_pin=meta["const_pin"])
    a_bits, b_bits, c_bits = g["a"], g["b"], g["z"]  # packed F128 — witness_to_rows unpacks on device (8x less host transfer)
    z = g["z"]; stmt = g["stmt"]

    def prove_once():
        root, codeword, tree = pcs_commit.commit(z, m, lir, lbs, mul=MUL, use_host_sha=True)
        ch = Challenger(b"flock-blake3-v0")
        prover.bind_statement(ch, stmt, root)
        zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, mul=MUL, ch=ch)
        x_ab = {"z_skip": zc["z"], "x_inner_rest": zc["mlv_challenges"][:ir],
                "x_outer": zc["mlv_challenges"][ir:]}
        _r, _zp, lc_claim, _zv = lincheck.prove(g["zlc"], None, None, x_ab, m, k_log, k_skip,
                                                mul=MUL, ch=ch, capture=True, circuit=csc)
        ab_full = np.concatenate([lc_claim["r_inner_rest"], x_ab["x_outer"]], axis=0)
        c_full = np.concatenate([zc["r_rest"][:ir], zc["r_rest"][ir:]], axis=0)
        return prover.open_batch(z, codeword, tree, [ab_full, c_full], k_code, lir, lbs, ch,
                                 mul=MUL, use_host_sha=True)

    t = best(prove_once, n=3)
    sp = f"{cpu/t:.1f}x vs same-instance flock BaseFold CPU {cpu:.0f}ms" if cpu else "(pass CPU ms as argv[1])"
    print(f"  GPU blake3 prove (commit+zerocheck+CSC-lincheck+open) {t:8.2f} ms   {sp}")


if __name__ == "__main__":
    main()
