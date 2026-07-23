"""Sustained host∥GPU pipelined throughput — witness generation overlapped with
the prove (flock-zorch#163).

Consumes the witness-blob queue `examples/gen_blake3_witness_stream.rs`
produces (fresh fused witnesses, bounded depth) and runs the full GPU prove
(commit → zerocheck → CSC lincheck → batched Ligerito open) on each, measuring
the steady-state system rate with witness generation INCLUDED. The static
config/statement/matrices come from a template golden; every witness-dependent
input (`z`/`a`/`b` packed + `z_lincheck`) comes from the blob, so shapes are
constant and the prove compiles once.

The queue depth sampled before each pull names the binding side directly:
pinned at the bound = GPU-bound (host keeps it full), pinned at zero =
host-bound (GPU starves).

`--gpu-only` reuses the template golden's witness in a loop with no queue — the
same consumer loop minus the host, for the overlap-cost comparison.

Run (start the producer first, or in parallel):
    cargo run --release --example gen_blake3_witness_stream -- \
        16384 /dev/shm/flock_witq 40 4
    export CUDA_ROOT="$HOME/.local/cuda13-merged"
    export FRX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PATH="$HOME/.local/cuda13/bin:$PATH"
    PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
        python/flock_zorch/testing/pipelined_prove_bench.py \
        --golden blake3_ligerito_golden_m28.bin [--queue /dev/shm/flock_witq]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402

from flock_zorch import lincheck, prover, zerocheck  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.testing import blake3_ligerito_oracle_test as blake3_gate  # noqa: E402
from flock_zorch.testing._golden import R  # noqa: E402
from flock_zorch.testing._util import await_all  # noqa: E402

DOMAIN = b"flock-blake3-lig-v0"


def read_blob(path: Path):
    """Parse one producer blob -> (z, a, b, zlc); the wire mirrors the goldens."""
    rd = R(path.read_bytes())
    assert bytes(rd.take(8)) == b"FLKWS_01", f"bad magic in {path}"
    m = rd.u()
    z, a, b = rd.fv(), rd.fv(), rd.fv()
    zlc = bytes(rd.raw(rd.u()))
    return m, z, a, b, zlc


def next_blob(queue: Path, timeout_s: float = 120.0) -> Path:
    """Oldest pending blob, waiting up to `timeout_s` for the producer."""
    t0 = time.perf_counter()
    while True:
        blobs = sorted(queue.glob("wit_*.bin"))
        if blobs:
            return blobs[0]
        if time.perf_counter() - t0 > timeout_s:
            raise SystemExit(f"queue {queue} empty for {timeout_s:.0f} s — producer dead?")
        time.sleep(0.002)


def make_prove(g):
    """The blake3 Ligerito prove as one closure over the static template fields;
    witness-dependent inputs are passed per call. Mirrors the byte gate's
    sequence (blake3_ligerito_oracle_test.run) minus the golden comparisons."""
    meta, cfg = g["meta"], g["cfg"]
    m, k_log, k_skip = meta["m"], meta["k_log"], meta["k_skip"]
    ir = k_log - k_skip
    stmt = g["stmt"]
    csc = lincheck.CscCircuit(g["a0_rows"], g["b0_rows"], 1 << k_log,
                              const_pin=meta["const_pin"])

    def prove(z, a, b, zlc):
        root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, z)
        ch = Challenger(DOMAIN)
        prover.bind_statement(ch, stmt, root)
        zc = zerocheck.prove_packed(a, b, z, m, ch=ch)
        x_ab = lincheck.AbClaimPoint.from_zerocheck(zc, ir)
        _lr, _zp, lc_claim = lincheck.prove(zlc, None, None, x_ab, m, k_log,
                                            k_skip, ch=ch, circuit=csc)
        ab_full = fnp.concatenate([lc_claim.r_inner_rest, x_ab.x_outer], axis=0)
        c_full = fnp.concatenate([zc.r_rest[:ir], zc.r_rest[ir:]], axis=0)
        return prover.open_batch_ligerito(cfg, z, pdata, [ab_full, c_full], ch)

    return prove


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", default="blake3_ligerito_golden.bin",
                    help="template golden under artifacts/ (static config/stmt/matrices)")
    ap.add_argument("--queue", default="/dev/shm/flock_witq")
    ap.add_argument("--proofs", type=int, default=40, help="total proofs to run")
    ap.add_argument("--warmup", type=int, default=3,
                    help="proofs excluded from the steady-state window (compile + warm)")
    ap.add_argument("--gpu-only", action="store_true",
                    help="no queue: loop the template witness (overlap-cost baseline)")
    args = ap.parse_args()

    print(f"device {frx.devices()[0]}")
    g = blake3_gate.load(args.golden)
    meta = g["meta"]
    n_hash = 1 << (meta["m"] - meta["k_log"])
    prove = make_prove(g)
    queue = Path(args.queue)

    depths, walls = [], []
    t_win = None
    for i in range(args.proofs):
        if args.gpu_only:
            depth = -1
            zi, ai, bi, zlci = g["z"], g["a"], g["b"], g["zlc"]
        else:
            depth = len(list(queue.glob("wit_*.bin")))
            blob = next_blob(queue)
            m_blob, zi, ai, bi, zlci = read_blob(blob)
            assert m_blob == meta["m"], f"blob m={m_blob} != template m={meta['m']}"
            blob.unlink()  # consumed — frees the producer's bounded slot
        zi, ai, bi = (frx.device_put(x) for x in (zi, ai, bi))

        t0 = time.perf_counter()
        await_all(prove(zi, ai, bi, zlci))
        wall = time.perf_counter() - t0
        depths.append(depth)
        walls.append(wall)
        print(f"proof {i:03}: {wall * 1e3:7.1f} ms  queue-depth-before-pull {depth}",
              flush=True)
        if i + 1 == args.warmup:
            t_win = time.perf_counter()

    n_win = args.proofs - args.warmup
    win_s = time.perf_counter() - t_win
    rate = n_win * n_hash / win_s
    tag = "GPU-only (no host)" if args.gpu_only else "pipelined (witness INCLUDED)"
    print(f"\n{tag}: {n_win} proofs x {n_hash} hashes in {win_s:.1f} s "
          f"= {rate / 1e3:.1f}K blake3/s sustained")
    print(f"  per-proof wall: median {np.median(walls[args.warmup:]) * 1e3:.1f} ms, "
          f"best {min(walls[args.warmup:]) * 1e3:.1f} ms")
    if not args.gpu_only:
        w = depths[args.warmup:]
        print(f"  queue depth before pull: min {min(w)} / median {int(np.median(w))} / "
              f"max {max(w)} — 0 means host-bound, the bound means GPU-bound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
