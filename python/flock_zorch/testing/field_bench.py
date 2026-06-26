"""Throughput baseline for the GF(2^128) GHASH multiply on GPU.

Establishes the perf number the zkx optimization must beat. The current
`field.mul` is a readable 64-step carryless bit-loop -- correct, slow, and (as
this bench shows) it does not fuse at scale: XLA materializes the long unrolled
XOR chain, so memory blows up well before bandwidth saturates. Reports
multiplies/sec, with XOR-add as the memory-bandwidth ceiling and the zk_dtypes
native binary_field_t7 multiply (TOWER basis, perf-only) for comparison.

Each section is guarded so one OOM does not kill the rest.
"""
import gc
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flock_zorch import field  # noqa: E402


def _rand(n, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.integers(0, 2**64, size=(n, 2), dtype=np.uint64))


def _bench(fn, a, b, iters):
    r = fn(a, b)
    r.block_until_ready()  # compile + warm
    t0 = time.perf_counter()
    for _ in range(iters):
        r = fn(a, b)
    r.block_until_ready()
    return (time.perf_counter() - t0) / iters


def _sweep(name, fn, logs, iters_fn, fmt):
    print(f"\n--- {name} ---")
    for log in logs:
        n = 1 << log
        try:
            a, b = _rand(n, 1), _rand(n, 2)
            dt = _bench(fn, a, b, iters_fn(log))
            print("  " + fmt(log, n, dt))
            del a, b
            gc.collect()
        except Exception as e:  # noqa: BLE001
            print(f"  n=2^{log:<2} FAILED: {type(e).__name__}: {str(e)[:110]}")
            if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
                break


def main():
    print("device:", jax.devices()[0], "| backend:", jax.default_backend())
    mul = jax.jit(field.mul)
    add = jax.jit(field.add)

    _sweep(
        "flock ghash_mul (GHASH basis, byte-exact)", mul,
        (16, 18, 20, 21, 22, 23, 24),
        lambda log: 100 if log <= 20 else 30,
        lambda log, n, dt: f"n=2^{log:<2} {dt*1e3:9.3f} ms  {n/dt/1e9:7.3f} G mul/s  "
                           f"{n*16/dt/1e9:6.1f} GB/s-out",
    )
    _sweep(
        "XOR add (memory-bandwidth ceiling)", add,
        (22, 24),
        lambda log: 50,
        lambda log, n, dt: f"n=2^{log:<2} {dt*1e3:9.3f} ms  {n/dt/1e9:7.3f} G add/s  "
                           f"{n*48/dt/1e9:6.1f} GB/s",
    )

    print("\n--- zk_dtypes binary_field_t7 NATIVE multiply (tower basis, perf-only) ---")
    try:
        import zk_dtypes
        t7 = zk_dtypes.binary_field_t7
        nmul = jax.jit(lambda x, y: x * y)
        for log in (20, 22, 24):
            n = 1 << log
            rng = np.random.default_rng(7)
            try:
                a = jnp.asarray(np.frombuffer(rng.bytes(n * 16), dtype=t7).copy())
                b = jnp.asarray(np.frombuffer(rng.bytes(n * 16), dtype=t7).copy())
                dt = _bench(nmul, a, b, 30)
                print(f"  n=2^{log:<2} {dt*1e3:9.3f} ms  {n/dt/1e9:7.3f} G mul/s")
                del a, b
                gc.collect()
            except Exception as e:  # noqa: BLE001
                print(f"  n=2^{log:<2} FAILED: {type(e).__name__}: {str(e)[:140]}")
    except Exception as e:  # noqa: BLE001
        print(f"  native t7 unavailable: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
