"""SHA-256 byte-match gate + CPU-vs-GPU benchmark.

(1) Loads flock's golden (`Sha256::digest` over N random messages) and asserts the
    frx port reproduces every 32-byte digest bit-for-bit — the oracle gate.
(2) Benchmarks batched SHA-256: GPU (one data-parallel call over all messages) vs
    flock's CPU `merkle::hash_leaf` over the same N messages (rayon-parallel, the
    honest flock baseline), and reports the speedup.

Run:
  cargo run --release --example dump_sha256 -- 65536 64 artifacts/sha256_golden.bin
  JAX_PLATFORMS=cuda PYTHONPATH=python <venv> python/flock_zorch/hash/testing/sha256_oracle_test.py
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import frx
import frx.numpy as jnp

from zorch.hash import sha256

REPO = Path(__file__).resolve().parents[4]
ART = REPO / "artifacts"
GPU_ITERS = 50


def _load_golden():
    raw = (ART / "sha256_golden.bin").read_bytes()
    assert raw[:8] == b"FLKSHA01", "bad magic"
    n = int.from_bytes(raw[8:16], "little")
    l = int.from_bytes(raw[16:24], "little")
    off = 24
    inp = np.frombuffer(raw, np.uint8, n * l, off).reshape(n, l)
    dig = np.frombuffer(raw, np.uint8, n * 32, off + n * l).reshape(n, 32)
    return n, l, inp, dig


def test_oracle():
    n, l, inp, golden = _load_golden()
    blocks = jnp.asarray(sha256._pad(inp))
    got = np.asarray(sha256.sha256_chain(sha256.INITIAL_STATE, blocks))
    assert got.shape == golden.shape, (got.shape, golden.shape)
    assert np.array_equal(got, golden), "sha256.digest != flock golden"
    return n, l


def _cpu_hash_ms(n: int, l: int) -> float:
    """Best-of-N ms for flock hashing N l-byte leaves (rayon merkle leaf level)."""
    out = subprocess.run(
        [str(REPO / "target" / "release" / "examples" / "bench_sha256_cpu"), str(n), str(l), "8"],
        capture_output=True, text=True, check=True,
    ).stdout
    line = next(ln for ln in out.splitlines() if ln.startswith("SHACPU"))
    return float(line.split()[3])  # best_ms


def main() -> int:
    n, l = test_oracle()
    print(f"device: {frx.devices()[0]} | backend: {frx.default_backend()}")
    print(f"SHA-256 byte-identity vs flock ({n} x {l}-byte msgs): PASS\n")

    inp = _load_golden()[2]
    blocks = jnp.asarray(sha256._pad(inp))
    fn = lambda b: sha256.sha256_chain(sha256.INITIAL_STATE, b)
    r = fn(blocks); r.block_until_ready()
    best = float("inf")
    for _ in range(GPU_ITERS):
        t0 = time.perf_counter()
        r = fn(blocks); r.block_until_ready()
        best = min(best, time.perf_counter() - t0)
    gpu_ms = best * 1e3

    # NOTE: byte-identity above IS the gate for this layer. SHA-256 throughput is
    # reported for context only — it is NOT a 10x target. The CPU has dedicated
    # SHA-256 hardware (SHA-NI) + 16 cores, so it wins here; but Merkle/SHA is
    # <1% of PCS-commit time (the NTT dominates 96-322x at m>=24, see README), so
    # this does not affect the end-to-end GPU prover speedup. In the fused prover
    # the tiny Merkle can stay on the host.
    try:
        cpu_ms = _cpu_hash_ms(n, l)
        ratio = gpu_ms / cpu_ms
        print(f"{'msgs':>8}  {'CPU flock ms':>13}  {'GPU zorch ms':>13}")
        print(f"{n:>8}  {cpu_ms:>13.3f}  {gpu_ms:>13.3f}   (GPU/CPU {ratio:.1f}x; "
              f"CPU SHA-NI wins — expected, and negligible in-prover)")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"(CPU bench unavailable: {type(e).__name__}; GPU = {gpu_ms:.3f} ms)")
    return 0  # gate = byte-identity (asserted above)


if __name__ == "__main__":
    sys.exit(main())
