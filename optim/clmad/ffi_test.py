"""Validate + benchmark the clmad GHASH multiply called from jax via XLA FFI.

Registers libghash_clmad.so's handler, calls it inside a jitted jax function, and
byte-compares against flock's golden (the now-retired `field_mul_golden.bin` — see
README Status). Standalone FFI path for the binary-field multiply:
jax binary-field mul -> ffi_call -> clmad kernel, no zkx rebuild.
"""
import ctypes
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

_lib = ctypes.cdll.LoadLibrary(str(HERE / "libghash_clmad.so"))
jax.ffi.register_ffi_target("flock_ghash_mul", jax.ffi.pycapsule(_lib.GhashMul), platform="CUDA")


@jax.jit
def ghash_mul(a, b):
    return jax.ffi.ffi_call("flock_ghash_mul", jax.ShapeDtypeStruct(a.shape, a.dtype))(a, b)


def load_golden(path):
    raw = path.read_bytes()
    assert raw[:8] == b"FLKMUL01"
    n = int.from_bytes(raw[8:16], "little")
    off, blk = 16, n * 16
    a = np.frombuffer(raw, np.uint64, n * 2, off).reshape(n, 2)
    b = np.frombuffer(raw, np.uint64, n * 2, off + blk).reshape(n, 2)
    o = np.frombuffer(raw, np.uint64, n * 2, off + 2 * blk).reshape(n, 2)
    return n, a, b, o


def main():
    print("backend:", jax.default_backend())
    n, a, b, golden = load_golden(REPO / "artifacts" / "field_mul_golden.bin")
    got = np.asarray(ghash_mul(jnp.asarray(a), jnp.asarray(b)))
    ok = np.array_equal(got, golden)
    print(f"clmad-FFI == flock golden ({n} pairs): {'PASS' if ok else 'FAIL'}")
    if not ok:
        i = int(np.flatnonzero(np.any(got != golden, axis=1))[0])
        print(f"  first mismatch @ {i}: got={got[i].tolist()} golden={golden[i].tolist()}")
        return

    N = 1 << 23
    ra = jnp.asarray(np.random.default_rng(1).integers(0, 2**64, size=(N, 2), dtype=np.uint64))
    rb = jnp.asarray(np.random.default_rng(2).integers(0, 2**64, size=(N, 2), dtype=np.uint64))
    r = ghash_mul(ra, rb)
    r.block_until_ready()
    it = 300
    t0 = time.perf_counter()
    for _ in range(it):
        r = ghash_mul(ra, rb)
    r.block_until_ready()
    dt = (time.perf_counter() - t0) / it
    print(f"clmad-FFI ghash_mul: {N/dt/1e9:.3f} G mul/s "
          f"({dt*1e3:.3f} ms @ N=2^23) — {N/dt/1e9/0.122:.0f}x vs software fori_loop")


if __name__ == "__main__":
    main()
