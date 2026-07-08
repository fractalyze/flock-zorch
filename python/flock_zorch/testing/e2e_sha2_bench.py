"""GPU SHA-256 prover timing (commit→zerocheck→CSC lincheck→batched open) vs
flock's BaseFold `prove` on the SAME real sha2 R1CS.

Ingests the dump_sha2 golden (witness + a=A·z + matrices are host setup), then
times the GPU prover phases on one shared challenger. Byte-identity is pinned by
sha2_oracle_test. CPU baseline: bench_sha2_cpu.rs (same instance, BaseFold).

HONEST scope: the timed GPU number is the PROVER (commit/zerocheck/lincheck/open);
the host matvec a=A·z (flock witness-prep, ~2%) is excluded. The CSC lincheck fold
now runs ON DEVICE (lincheck.CscCircuit sorted prefix-XOR scan, not the old host
np.bitwise_xor.at).

Run:
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/e2e_sha2_bench.py
"""
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from flock_zorch import field, pcs_commit, zerocheck, lincheck, prover  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.testing.sha2_oracle_test import load, _unpack  # noqa: E402
from flock_zorch.testing._util import best  # noqa: E402

ART = Path(__file__).resolve().parents[3] / "artifacts"
# flock BaseFold prove on the same instance (bench_sha2_cpu.rs), x86 scalar.
CPU_SHA2 = {18: None, 21: None}  # filled in by the runner from bench_sha2_cpu output


def main():
    cpu = None
    if len(sys.argv) > 1:
        cpu = float(sys.argv[1])  # optional: flock BaseFold CPU ms for this instance
    _, g = load()
    meta = g["meta"]; m = meta["m"]; lir = meta["lir"]; lbs = meta["lbs"]
    k_log = meta["k_log"]; k_skip = meta["k_skip"]; ir = k_log - k_skip
    k_code = (m - 7 - lbs) + lir
    print(f"device {jax.devices()[0]} | sha2 m={m}")

    # setup (host witness-prep / circuit build — not the GPU prover, not timed)
    csc = lincheck.CscCircuit(g["a0_rows"], g["b0_rows"], 1 << k_log, const_pin=meta["const_pin"])
    a_bits, b_bits, c_bits = g["a"], g["b"], g["z"]  # packed F128 — witness_to_rows unpacks on device (8x less host transfer)
    z = g["z"]; stmt = g["stmt"]

    def prove_once():
        root, codeword, tree = pcs_commit.commit(z, m, lir, lbs)
        ch = Challenger(b"flock-sha2-v0")
        prover.bind_statement(ch, stmt, root)
        zc = zerocheck.prove_packed(a_bits, b_bits, c_bits, m, ch=ch)
        x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
        _r, _zp, lc_claim, _zv = lincheck.prove(g["zlc"], None, None, x_ab, m, k_log, k_skip, ch=ch, capture=True, circuit=csc)
        ab_full = np.concatenate([lc_claim.r_inner_rest, x_ab.x_outer], axis=0)
        c_full = np.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
        return prover.open_batch(z, codeword, tree, [ab_full, c_full], k_code, lir, lbs, ch)

    t = best(prove_once, n=3)
    sp = f"{cpu/t:.1f}x vs same-instance flock BaseFold CPU {cpu:.0f}ms" if cpu else "(pass CPU ms as argv[1])"
    print(f"  GPU sha256 prove (commit+zerocheck+CSC-lincheck+open) {t:8.2f} ms   {sp}")


if __name__ == "__main__":
    main()
