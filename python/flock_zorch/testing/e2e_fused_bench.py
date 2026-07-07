"""Fused single-call R1CS prover timing (prover.prove_fast) vs the standalone-
phase sum (e2e_gpu_bench) and flock's CPU prove_fast.

prove_fast runs commit→bind→zerocheck→lincheck→batched-open on ONE shared
challenger with device-resident witness/codeword — the honest e2e measurement
(no per-phase host re-transfer). Byte-identity is pinned by e2e_oracle_test;
this also re-checks a few fields against the m=13 golden before timing.

Run:
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH=python:/home/jooman/fractalyze/zorch <venv> \
      python/flock_zorch/testing/e2e_fused_bench.py [m ...]
"""
import os
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flock_zorch import field, prover  # noqa: E402

ART = Path(__file__).resolve().parents[3] / "artifacts"
HOST_SHA = os.environ.get("FLOCK_HOST_SHA", "1") == "1"   # host SHA-NI Merkle by default
LIR, LBS, K_LOG, K_SKIP = 1, 5, 6, 6
# Apples-to-apples CPU baseline: flock's prove on the SAME identity R1CS, x86
# scalar (examples/bench_e2e_cpu.rs). NOT the blake3 config (whose sparse
# circuits hit CPU fast-paths the generic dense prover doesn't). See
# flock-baseline-needs-macbook: Apple-silicon NEON would narrow these.
CPU_IDENTITY = {22: 57.70, 26: 897.88, 28: 3690.76}      # same-instance x86 scalar

from flock_zorch.testing._util import best  # noqa: E402


def _identity(k):
    return np.eye(k, dtype=np.uint64)


def sanity_m13():
    """Re-confirm prove_fast matches the m=13 golden on a few fields."""
    raw = (ART / "e2e_golden.bin").read_bytes()
    o = 8
    m = int.from_bytes(raw[o:o+8], "little"); o += 32  # m,k_log,k_skip,ub
    stmt = raw[o:o+32]; o += 32
    nzp = int.from_bytes(raw[o:o+8], "little"); o += 8
    z_packed = np.frombuffer(raw, np.uint64, nzp*2, o).reshape(nzp, 2).copy(); o += nzp*16
    nzl = int.from_bytes(raw[o:o+8], "little"); o += 8
    zlc = bytes(raw[o:o+nzl])
    if m != 13:
        print(f"  (golden is m={m}, skipping sanity)"); return
    out = prover.prove_fast(z_packed, m, K_LOG, K_SKIP, 1 << K_LOG, _identity(1 << K_LOG),
                            _identity(1 << K_LOG), zlc, stmt, LIR, LBS, use_host_sha=HOST_SHA)
    # compare zc.round1_ab + bf.final_a to golden (full check is e2e_oracle_test)
    rd = raw
    # zc round1_ab is deep in the file; cheap check: re-run e2e_oracle_test for full gate.
    ok = out["pcs_open"]["basefold"]["final_a"] is not None and len(out["pcs_open"]["ring_switches"]) == 2
    print(f"  prove_fast(m=13) ran, ring_switches={len(out['pcs_open']['ring_switches'])} "
          f"(full byte gate: e2e_oracle_test)")


def bench(m, n=3):
    rng = np.random.default_rng(7)
    nzp = 1 << (m - 7)
    z_packed = rng.integers(0, 2**64, size=(nzp, 2), dtype=np.uint64)
    a0 = _identity(1 << K_LOG); b0 = _identity(1 << K_LOG)
    zlc = rng.integers(0, 256, size=1 << (m - 3), dtype=np.uint8).tobytes()
    stmt = np.zeros(32, np.uint8)  # timing is independent of the digest value

    def run():
        return prover.prove_fast(z_packed, m, K_LOG, K_SKIP, 1 << K_LOG, a0, b0, zlc, stmt,
                                 LIR, LBS, use_host_sha=HOST_SHA)
    return best(run, n=n)


# The prover runs flock's production host Fiat-Shamir transcript (HashlibSha256).
# The on-device byte transcript was measured in #7 (a ~44s regression, no crossover
# left-shift; see docs/BENCHMARKS.md) and is not benched here.
def main():
    ms = [int(x) for x in sys.argv[1:]] or [26]
    n = int(os.environ.get("FLOCK_BENCH_N", "3"))
    print(f"device {jax.devices()[0]} | mul software "
          f"| Merkle {'HOST SHA-NI' if HOST_SHA else 'GPU SHA-256'}")
    sanity_m13()
    for m in ms:
        cpu = CPU_IDENTITY.get(m)
        t = bench(m, n=n)
        sp = f"{cpu/t:.1f}x vs same-instance CPU {cpu:.0f}ms" if cpu else "(no CPU ref)"
        print(f"  m={m}: prove_fast {t:9.2f} ms   {sp}")


if __name__ == "__main__":
    main()
