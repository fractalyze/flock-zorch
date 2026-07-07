"""Assemble + validate + benchmark the clmad GHASH multiply (optimization #1).

Byte-compares against flock's golden (`artifacts/field_mul_golden.bin`) and reports
throughput vs the software baseline. Needs CUDA 13.x ptxas at ~/.local/cuda13 (see
README) and the RTX 5090. Paths resolve relative to this file.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import cuda_run as cr  # noqa: E402

PTXAS = os.path.expanduser("~/.local/cuda13/bin/ptxas")
PTX = HERE / "ghash_mul.ptx"
CUBIN = Path("/tmp") / "flock_ghash_clmad.cubin"
GOLDEN = REPO / "artifacts" / "field_mul_golden.bin"


def load_golden(path):
    raw = path.read_bytes()
    assert raw[:8] == b"FLKMUL01"
    n = int.from_bytes(raw[8:16], "little")
    off, blk = 16, n * 16
    return n, raw[off:off + blk], raw[off + blk:off + 2 * blk], raw[off + 2 * blk:off + 3 * blk]


def main():
    if not os.path.exists(PTXAS):
        sys.exit(f"need CUDA 13.x ptxas at {PTXAS} (see README)")
    if not GOLDEN.exists():
        sys.exit(f"missing {GOLDEN}: dump_field_mul was retired in #44/#45 — "
                 "regenerate a reference (see README Status)")
    subprocess.run([PTXAS, "-arch=sm_120", "-O3", str(PTX), "-o", str(CUBIN)], check=True)

    cr.init()
    f = cr.func(cr.load(str(CUBIN)), "ghash_mul")
    block = 256

    n, a_b, b_b, golden = load_golden(GOLDEN)
    da, db, do = cr.alloc(n * 16), cr.alloc(n * 16), cr.alloc(n * 16)
    cr.htod(da, a_b)
    cr.htod(db, b_b)
    cr.launch(f, (n + block - 1) // block, block, [da, db, do, cr._u32(n)])
    cr.sync()
    ok = cr.dtoh(do, n * 16) == golden
    print(f"clmad == flock golden ({n} pairs): {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)

    N = 1 << 23
    ra = np.random.default_rng(1).integers(0, 2**64, size=(N, 2), dtype=np.uint64).tobytes()
    rb = np.random.default_rng(2).integers(0, 2**64, size=(N, 2), dtype=np.uint64).tobytes()
    da2, db2, do2 = cr.alloc(N * 16), cr.alloc(N * 16), cr.alloc(N * 16)
    cr.htod(da2, ra)
    cr.htod(db2, rb)
    grid = (N + block - 1) // block
    cr.launch(f, grid, block, [da2, db2, do2, cr._u32(N)])
    cr.sync()
    it = 300
    t0 = time.perf_counter()
    for _ in range(it):
        cr.launch(f, grid, block, [da2, db2, do2, cr._u32(N)])
    cr.sync()
    dt = (time.perf_counter() - t0) / it
    print(f"clmad ghash_mul: {N/dt/1e9:.3f} G mul/s "
          f"({dt*1e3:.3f} ms @ N=2^23, {N*48/dt/1e9:.0f} GB/s) — {N/dt/1e9/0.122:.0f}x vs software")


if __name__ == "__main__":
    main()
