"""GPU-vs-CPU 10x gate for the sumcheck eq-table build — the dominant
data-parallel primitive of flock's multilinear sumcheck (iter 10).

Anchors flock-zorch's GPU `build_eq` (clmad) against **unmodified succinct flock**
on the SAME x86 box, byte-identical:

  1. byte-identity: the jax `build_eq` reproduces flock's reference bytes on the
     dumped fixture (the oracle gate, reused from `sumcheck_oracle_test`);
  2. speed: for each size, time flock's CPU `build_eq` (via the
     `bench_sumcheck_cpu` example) and the GPU clmad `build_eq`, best-of-N each,
     and report the speedup. Fails if any size is under 10x.

CPU baseline caveat: flock's eq build is its scalar reference — the only path
that compiles on x86. flock is tuned for Apple silicon (NEON, aarch64-gated), so
a true apples-to-apples comparison needs flock built on a MacBook; see the
project memory `flock-baseline-needs-macbook`.

Run:
  cargo build --release --example bench_sumcheck_cpu        # once
  export PATH="$HOME/.local/cuda13/bin:$PATH"               # clmad cubin assembler
  JAX_PLATFORMS=cuda PYTHONPATH=python <venv> \
      python/flock_zorch/testing/sumcheck_gpu_vs_cpu.py
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flock_zorch import field, sumcheck  # noqa: E402
from flock_zorch.testing import sumcheck_oracle_test as oracle  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SIZES = (16, 18, 20)
CPU_ITERS = 8
GPU_ITERS = 50
TARGET = 10.0


def _cpu_eq_ms(n: int) -> float:
    """Best-of-N ms for flock's CPU build_eq at length n (output 2^n)."""
    deps = REPO / "target" / "release" / "examples"
    cand = [c for c in sorted(deps.glob("bench_sumcheck_cpu*")) if c.is_file() and c.suffix == ""]
    if cand:
        out = subprocess.run(
            [str(cand[0]), str(n), str(CPU_ITERS)],
            capture_output=True, text=True, check=True,
        ).stdout
    else:
        out = subprocess.run(
            ["cargo", "run", "--quiet", "--release", "--example", "bench_sumcheck_cpu",
             "--", str(n), str(CPU_ITERS)],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
    line = next(ln for ln in out.splitlines() if ln.startswith("EQCPU"))
    return float(line.split()[2])  # best_ms


def _gpu_eq_ms(fn, r) -> float:
    out = fn(r)
    out.block_until_ready()
    best = float("inf")
    for _ in range(GPU_ITERS):
        t0 = time.perf_counter()
        out = fn(r)
        out.block_until_ready()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main() -> int:
    mul = field.mul
    print(f"device: {jax.devices()[0]} | backend: {jax.default_backend()} | "
          "mul: software loop")
    print("CPU baseline: unmodified flock build_eq (x86 scalar; flock's NEON is "
          "aarch64-gated — see flock-baseline-needs-macbook)\n")

    # 1. byte-identity gate (reuse the oracle on the dumped fixture).
    oracle.run(mul=mul)
    print("byte-identity vs flock (build_eq / round_pair / fold_single): PASS")

    # 2. speed.
    print(f"\n{'n':>3}  {'2^n elems':>12}  {'CPU flock ms':>13}  {'GPU zorch ms':>13}  {'speedup':>9}")
    worst = float("inf")
    for n in SIZES:
        r = jnp.asarray(np.random.default_rng(7).integers(0, 2**64, size=(n, 2), dtype=np.uint64))
        fn = jax.jit(lambda rr: sumcheck.build_eq(rr, mul=mul))
        gpu = _gpu_eq_ms(fn, r)
        cpu = _cpu_eq_ms(n)
        spd = cpu / gpu
        worst = min(worst, spd)
        print(f"{n:>3}  {1 << n:>12}  {cpu:>13.3f}  {gpu:>13.3f}  {spd:>8.1f}x")

    print(f"\nworst-case speedup: {worst:.1f}x  (target >= {TARGET:.0f}x)")
    if worst >= TARGET:
        print(f"GATE PASS: GPU build_eq is >= {TARGET:.0f}x faster than CPU flock on every size.")
        return 0
    print(f"GATE FAIL: worst case {worst:.1f}x < {TARGET:.0f}x.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
