"""Consolidated per-layer benchmark for flock-zorch — run each iteration to track
improvement. Reports field-mul (current fori_loop vs the original unroll),
additive-NTT, the sumcheck core (build_eq / round_pair / fold), and the XOR-add
bandwidth ceiling on the active backend.
"""
import gc
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jax import lax  # noqa: E402
from zorch.coding.additive_reed_solomon import AdditiveReedSolomon  # noqa: E402

from flock_zorch import field, sumcheck  # noqa: E402

U64 = jnp.uint64
_ONE = U64(1)


# --- iter-2 baseline: the original Python-unrolled clmul, kept for A/B only ---
def _clmul64_unroll(a, b):
    lo = jnp.zeros_like(a)
    hi = jnp.zeros_like(a)
    for i in range(64):
        mask = U64(0) - ((a >> U64(i)) & _ONE)
        lo = lo ^ (mask & (b << U64(i)))
        if i != 0:
            hi = hi ^ (mask & (b >> U64(64 - i)))
    return lo, hi


def _mul_unroll(a, b):
    alo, ahi = a[..., 0], a[..., 1]
    blo, bhi = b[..., 0], b[..., 1]
    ll = _clmul64_unroll(alo, blo)
    lh = _clmul64_unroll(alo, bhi)
    hl = _clmul64_unroll(ahi, blo)
    hh = _clmul64_unroll(ahi, bhi)
    r0 = ll[0]
    r1 = ll[1] ^ lh[0] ^ hl[0]
    r2 = hh[0] ^ lh[1] ^ hl[1]
    r3 = hh[1]
    lo, hi = field._ghash_reduce(r0, r1, r2, r3)
    return jnp.stack([lo, hi], axis=-1)


def _rand(n, seed):
    return jnp.asarray(np.random.default_rng(seed).integers(0, 2**64, size=(n, 2), dtype=np.uint64))


def _bench(fn, args, iters):
    r = fn(*args)
    jax.block_until_ready(r)
    t0 = time.perf_counter()
    for _ in range(iters):
        r = fn(*args)
    jax.block_until_ready(r)
    return (time.perf_counter() - t0) / iters


def _oom(e):
    return "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e)


def main():
    print("device:", jax.devices()[0], "| backend:", jax.default_backend())
    mul_loop = jax.jit(field.mul)
    mul_unroll = jax.jit(_mul_unroll)
    add = jax.jit(field.add)

    print("\n[field mul]  G mul/s   (iter2 unroll  vs  iter3 fori_loop)")
    for log in (20, 22, 23):
        n = 1 << log
        a, b = _rand(n, 1), _rand(n, 2)
        line = f"  2^{log:<2}"
        for name, fn in (("unroll", mul_unroll), ("loop", mul_loop)):
            try:
                dt = _bench(fn, (a, b), 30)
                line += f"  {name}={n/dt/1e9:6.3f}"
            except Exception as e:  # noqa: BLE001
                line += f"  {name}={'OOM' if _oom(e) else type(e).__name__}"
        print(line)
        del a, b
        gc.collect()

    print("\n[additive NTT]  AdditiveReedSolomon.encode (blowup=1, ghash)")
    for log in (16, 18, 20):
        n = 1 << log
        d = _rand(n, 3)
        code = AdditiveReedSolomon(n, 1, jnp.binary_field_ghash)
        fn = jax.jit(lambda dd, c=code: c.encode(lax.bitcast_convert_type(dd, jnp.binary_field_ghash)))
        try:
            dt = _bench(fn, (d,), 20)
            print(f"  log_d={log:<2} {dt*1e3:8.2f} ms/transform  {n/dt/1e9:6.3f} G elem/s")
        except Exception as e:  # noqa: BLE001
            print(f"  log_d={log:<2} {'OOM' if _oom(e) else type(e).__name__}: {str(e)[:70]}")
        del d
        gc.collect()

    sc_mul = field.mul
    sc_tag = "software"
    print(f"\n[sumcheck core]  mul={sc_tag}")
    for log in (16, 18, 20):
        n = 1 << log
        r = _rand(log, 5)
        a, b = _rand(n, 6), _rand(n, 7)
        eq_fn = jax.jit(lambda rr, ln=log: sumcheck.build_eq(rr, mul=sc_mul))
        rp_fn = jax.jit(lambda aa, bb, rr: sumcheck.round_pair(aa, bb, rr, mul=sc_mul))
        fs_fn = jax.jit(lambda aa: sumcheck.fold_single(aa, r[0], mul=sc_mul))
        eq_ms = _bench(eq_fn, (r,), 30) * 1e3
        rp_ms = _bench(rp_fn, (a, b, r), 30) * 1e3
        fs_ms = _bench(fs_fn, (a,), 30) * 1e3
        print(f"  log={log:<2} build_eq {eq_ms:7.3f} ms ({n/(eq_ms/1e3)/1e9:5.2f} G elem/s)"
              f"  round_pair {rp_ms:7.3f} ms  fold {fs_ms:7.3f} ms")
        del a, b
        gc.collect()

    print("\n[XOR add]  bandwidth ceiling")
    n = 1 << 24
    a, b = _rand(n, 1), _rand(n, 2)
    dt = _bench(add, (a, b), 50)
    print(f"  2^24  {n/dt/1e9:6.1f} G add/s  {n*48/dt/1e9:.0f} GB/s")


if __name__ == "__main__":
    main()
