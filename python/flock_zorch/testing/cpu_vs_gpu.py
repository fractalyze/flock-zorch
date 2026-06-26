"""GPU-vs-CPU 10x gate for flock-zorch — the headline benchmark.

Anchors flock-zorch's GPU additive-NTT against **unmodified succinct flock** on
the SAME hardware and asserts the GPU is >=10x faster, byte-identical:

  1. byte-identity: run the jax NTT (clmad) on flock's golden fixture and assert
     it equals flock's reference output bit-for-bit (the oracle gate);
  2. speed: for each size, time flock's CPU `forward_transform_scalar` (via the
     `bench_ntt_cpu` example) and the GPU clmad NTT, best-of-N each, and report
     the speedup. Fails if any size is under 10x.

The CPU baseline is flock's software-scalar path — the only one that compiles on
x86 (flock's NEON/parallel paths are aarch64-gated). We note this in the report.

Run:
  cargo build --release --example bench_ntt_cpu          # once
  export PATH="$HOME/.local/cuda13/bin:$PATH"            # clmad cubin assembler
  JAX_PLATFORMS=cuda PYTHONPATH=python <venv> \
      python/flock_zorch/testing/cpu_vs_gpu.py
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flock_zorch import field, field_clmad, ntt as ntt_mod  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
ART = REPO / "artifacts"
SIZES = (16, 18, 20)
CPU_ITERS = 8
GPU_ITERS = 50
TARGET = 10.0


def _cpu_ntt_ms(log_d: int) -> float:
    """Best-of-N ms for flock's CPU forward_transform_scalar at 2^log_d."""
    binname = None
    deps = REPO / "target" / "release" / "examples"
    cand = sorted(deps.glob("bench_ntt_cpu*"))
    cand = [c for c in cand if c.is_file() and c.suffix == "" ]
    if cand:
        binname = cand[0]
    if binname is None:
        # fall back to `cargo run` (slower to start, but self-building)
        out = subprocess.run(
            ["cargo", "run", "--quiet", "--release", "--example", "bench_ntt_cpu",
             "--", str(log_d), str(CPU_ITERS)],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
    else:
        out = subprocess.run(
            [str(binname), str(log_d), str(CPU_ITERS)],
            capture_output=True, text=True, check=True,
        ).stdout
    line = next(ln for ln in out.splitlines() if ln.startswith("NTTCPU"))
    return float(line.split()[2])  # best_ms


def _gpu_ntt_ms(fn, d, tw) -> float:
    r = fn(d, tw)
    r.block_until_ready()
    best = float("inf")
    for _ in range(GPU_ITERS):
        t0 = time.perf_counter()
        r = fn(d, tw)
        r.block_until_ready()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _byte_gate(mul) -> None:
    raw = (ART / "ntt_golden.bin").read_bytes()
    log_d = int.from_bytes(raw[8:16], "little")
    n, ntw, off = 1 << log_d, (1 << log_d) - 1, 16
    inp = np.frombuffer(raw, np.uint64, n * 2, off).reshape(n, 2)
    tw = np.frombuffer(raw, np.uint64, ntw * 2, off + n * 16).reshape(ntw, 2)
    out = np.frombuffer(raw, np.uint64, n * 2, off + n * 16 + ntw * 16).reshape(n, 2)
    fn = jax.jit(lambda d, t: ntt_mod.forward_transform_scalar(d, t, log_d, mul=mul))
    got = np.asarray(fn(jnp.asarray(inp), jnp.asarray(tw)))
    assert np.array_equal(got, out), f"NTT byte-gate FAILED at log_d={log_d}"
    print(f"byte-identity vs flock @ log_d={log_d}: PASS")


def main() -> int:
    use_clmad = field_clmad.available()
    mul = field_clmad.mul if use_clmad else field.mul
    print(f"device: {jax.devices()[0]} | backend: {jax.default_backend()} | "
          f"mul: {'clmad (FFI)' if use_clmad else 'software loop'}")
    print("CPU baseline: unmodified flock forward_transform_scalar "
          "(x86 software path; NEON/parallel are aarch64-gated)\n")

    _byte_gate(mul)

    print(f"\n{'log_d':>5}  {'CPU flock ms':>13}  {'GPU zorch ms':>13}  {'speedup':>9}")
    worst = float("inf")
    rows = []
    for log in SIZES:
        n = 1 << log
        d = jnp.asarray(np.random.default_rng(3).integers(0, 2**64, size=(n, 2), dtype=np.uint64))
        tw = jnp.asarray(np.random.default_rng(4).integers(0, 2**64, size=(n - 1, 2), dtype=np.uint64))
        fn = jax.jit(lambda dd, tt, ld=log: ntt_mod.forward_transform_scalar(dd, tt, ld, mul=mul))
        gpu = _gpu_ntt_ms(fn, d, tw)
        cpu = _cpu_ntt_ms(log)
        spd = cpu / gpu
        worst = min(worst, spd)
        rows.append((log, cpu, gpu, spd))
        print(f"{log:>5}  {cpu:>13.3f}  {gpu:>13.3f}  {spd:>8.1f}x")

    print(f"\nworst-case speedup: {worst:.1f}x  (target >= {TARGET:.0f}x)")
    if worst >= TARGET:
        print(f"GATE PASS: GPU is >= {TARGET:.0f}x faster than CPU flock on every size.")
        return 0
    print(f"GATE FAIL: worst case {worst:.1f}x < {TARGET:.0f}x.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
