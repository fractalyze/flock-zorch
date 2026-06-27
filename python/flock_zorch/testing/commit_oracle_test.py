"""PCS commit byte-match gate + GPU-vs-CPU speedup (first full sub-protocol).

(1) Byte gate: load flock's golden (`pcs::commit` over a packed witness) and
    assert the jax port reproduces the 32-byte Merkle root bit-for-bit — this
    covers the WHOLE commit (pack-layout + interleaved NTT + SHA-256 Merkle).
(2) Speedup: time the commit's dominant compute — the interleaved forward NTT
    (RS encode) — on GPU vs flock's CPU `forward_transform_interleaved`, and
    report the ratio. Twiddles are data-independent (amortized once per param
    set), so they're precomputed outside the timed region on both sides. Merkle
    is the <1% CPU-favorable tail (reported, not gated on speed).

Run:
  cargo build --release --example dump_commit --example bench_commit_cpu
  ./target/release/examples/dump_commit 20 1 5 artifacts/commit_golden.bin
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda PYTHONPATH=python <venv> python/flock_zorch/testing/commit_oracle_test.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)  # uint64 field lanes
import jax.numpy as jnp  # noqa: E402

from flock_zorch import field, ntt as ntt_mod, pcs_commit  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
ART = REPO / "artifacts"
GPU_ITERS = 50
TARGET = 10.0
_HOST = os.environ.get("FLOCK_HOST_SHA") == "1"  # gate the host SHA-NI Merkle path too

try:
    from flock_zorch import field_clmad
    _MUL = field_clmad.mul if field_clmad.available() else field.mul
    _MULNAME = "clmad" if field_clmad.available() else "software"
except Exception:  # noqa: BLE001
    _MUL, _MULNAME = field.mul, "software"


def _load_golden():
    raw = (ART / "commit_golden.bin").read_bytes()
    assert raw[:8] == b"FLKCMT01", "bad magic"
    m = int.from_bytes(raw[8:16], "little")
    lir = int.from_bytes(raw[16:24], "little")
    lbs = int.from_bytes(raw[24:32], "little")
    n_packed = 1 << (m - 7)
    off = 32
    z_packed = np.frombuffer(raw, np.uint64, n_packed * 2, off).reshape(n_packed, 2)
    root = np.frombuffer(raw, np.uint8, 32, off + n_packed * 16)
    return m, lir, lbs, z_packed, root


def _cpu_commit_ms(m, lir, lbs):
    out = subprocess.run(
        [str(REPO / "target/release/examples/bench_commit_cpu"), str(m), str(lir), str(lbs), "8"],
        capture_output=True, text=True, check=True,
    ).stdout
    p = next(ln for ln in out.splitlines() if ln.startswith("CMTCPU")).split()
    return float(p[2]), float(p[3]), float(p[4])  # ntt, merkle, total (best ms)


def main() -> int:
    m, lir, lbs, z_packed, golden = _load_golden()
    print(f"device: {jax.devices()[0]} | backend: {jax.default_backend()} | mul: {_MULNAME}"
          f"{' | HOST SHA-NI Merkle' if _HOST else ''}")
    print(f"commit params: m={m} rate=1/2^{lir} batch=2^{lbs}\n")

    # (1) Byte gate over the full commit root.
    got = pcs_commit.commit_root(z_packed, m, lir, lbs, mul=_MUL, use_host_sha=_HOST)
    ok = np.array_equal(got, golden)
    print(f"PCS commit root byte-identity vs flock: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(" got :", bytes(got).hex())
        print(" want:", bytes(golden).hex())
        return 1

    # (2) Speedup on the dominant compute (interleaved forward NTT).
    log_msg, log_dim = m - 7, m - 7 - lbs
    k_code = log_dim + lir
    num_ntts, n_pos_code, n_pos_msg = 1 << lbs, 1 << k_code, 1 << log_dim
    x = jnp.asarray(z_packed).reshape(n_pos_msg, num_ntts, 2)
    pad = jnp.zeros((n_pos_code - n_pos_msg, num_ntts, 2), x.dtype)
    codeword = jnp.concatenate([x, pad], 0).reshape(n_pos_code * num_ntts, 2)
    tw = jnp.asarray(ntt_mod.compute_twiddles(k_code))  # amortized (data-independent)
    fn = jax.jit(lambda c, t: ntt_mod.forward_transform_interleaved(c, t, k_code, num_ntts, mul=_MUL))
    r = fn(codeword, tw); r.block_until_ready()
    best = float("inf")
    for _ in range(GPU_ITERS):
        t0 = time.perf_counter(); r = fn(codeword, tw); r.block_until_ready()
        best = min(best, time.perf_counter() - t0)
    gpu_ntt_ms = best * 1e3

    cpu_ntt, cpu_mrk, cpu_tot = _cpu_commit_ms(m, lir, lbs)
    spd = cpu_ntt / gpu_ntt_ms
    print(f"\n  commit encode (interleaved NTT):  CPU {cpu_ntt:.3f} ms  |  GPU {gpu_ntt_ms:.3f} ms"
          f"  =>  {spd:.1f}x")
    print(f"  commit Merkle (<1% tail, CPU SHA-NI): CPU {cpu_mrk:.3f} ms  (stays on host in fused prover)")
    print(f"\n{'GATE PASS' if spd >= TARGET else 'GATE FAIL'}: GPU commit encode is "
          f"{spd:.1f}x CPU flock (target >= {TARGET:.0f}x); root byte-identical.")
    return 0 if spd >= TARGET else 1


if __name__ == "__main__":
    sys.exit(main())
