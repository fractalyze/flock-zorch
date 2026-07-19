"""Consolidated per-layer benchmark for flock-zorch — run each iteration to track
improvement. Reports the additive-NTT, the sumcheck core (build_eq / round_pair /
fold), and the XOR-add bandwidth ceiling on the active backend. Field multiplies
now run on the native `binary_field_ghash` dtype (no software field to bench).
"""
import gc
import time

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)
import frx.numpy as fnp  # noqa: E402

from zorch.coding.additive_reed_solomon import AdditiveReedSolomon  # noqa: E402

from flock_zorch import sumcheck, ghash  # noqa: E402


def _rand(n, seed):
    return fnp.asarray(np.random.default_rng(seed).integers(0, 2**64, size=(n, 2), dtype=np.uint64))


def _bench(fn, args, iters):
    r = fn(*args)
    frx.block_until_ready(r)
    t0 = time.perf_counter()
    for _ in range(iters):
        r = fn(*args)
    frx.block_until_ready(r)
    return (time.perf_counter() - t0) / iters


def _oom(e):
    return "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e)


def main():
    print("device:", frx.devices()[0], "| backend:", frx.default_backend())

    print("\n[additive NTT]  AdditiveReedSolomon.encode (blowup=1, ghash)")
    for log in (16, 18, 20):
        n = 1 << log
        d = _rand(n, 3)
        code = AdditiveReedSolomon(n, 1, fnp.binary_field_ghash)
        fn = frx.jit(lambda dd, c=code: c.encode(ghash.to_ghash(dd)))
        try:
            dt = _bench(fn, (d,), 20)
            print(f"  log_d={log:<2} {dt*1e3:8.2f} ms/transform  {n/dt/1e9:6.3f} G elem/s")
        except Exception as e:  # noqa: BLE001
            print(f"  log_d={log:<2} {'OOM' if _oom(e) else type(e).__name__}: {str(e)[:70]}")
        del d
        gc.collect()

    print("\n[sumcheck core]")
    for log in (16, 18, 20):
        n = 1 << log
        r = _rand(log, 5)
        a, b = _rand(n, 6), _rand(n, 7)
        eq_fn = frx.jit(lambda rr, ln=log: sumcheck.build_eq_lanes(rr))
        rp_fn = frx.jit(lambda aa, bb, rr: sumcheck.round_pair_lanes(aa, bb, rr))
        fs_fn = frx.jit(lambda aa: sumcheck.fold_single(aa, r[0]))
        eq_ms = _bench(eq_fn, (r,), 30) * 1e3
        rp_ms = _bench(rp_fn, (a, b, r), 30) * 1e3
        fs_ms = _bench(fs_fn, (a,), 30) * 1e3
        print(f"  log={log:<2} build_eq {eq_ms:7.3f} ms ({n/(eq_ms/1e3)/1e9:5.2f} G elem/s)"
              f"  round_pair {rp_ms:7.3f} ms  fold {fs_ms:7.3f} ms")
        del a, b
        gc.collect()

    print("\n[XOR add]  bandwidth ceiling")
    add = frx.jit(lambda a, b: a ^ b)
    n = 1 << 24
    a, b = _rand(n, 1), _rand(n, 2)
    dt = _bench(add, (a, b), 50)
    print(f"  2^24  {n/dt/1e9:6.1f} G add/s  {n*48/dt/1e9:.0f} GB/s")


if __name__ == "__main__":
    main()
