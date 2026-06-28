"""clmad-accelerated GF(2^128) multiply via XLA FFI — byte-identical to
flock_zorch.field.mul, the memory-bound GPU fast path (PTX `clmad`; benchmarks in
optim/clmad/README.md).

Drop-in for `field.mul`: same uint64 [..., 2] contract, handles broadcasting. The
FFI handler (`optim/clmad/libghash_clmad.so`) launches the clmad cubin on XLA's
stream — no zkx rebuild. Build it with `optim/clmad/build_ffi.sh`; see that dir's
README. `add` is re-exported from `field` (XOR needs no acceleration). This is the
runtime analog of flock-core `gf2_128.rs`'s compile-time `#[cfg]` `Mul` dispatch
(NEON / vpclmulqdq / software); `field.mul` is the portable software arm.

Use `available()` to gate: it checks the built .so is present; the CUDA-13.3 cubin
+ sm_120 GPU are validated lazily on first `mul` (FFI registration + kernel
launch). On CPU or without the handler, use `flock_zorch.field` instead.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp

from flock_zorch.field import add  # noqa: F401  (re-export; XOR is already optimal)

_SO = Path(__file__).resolve().parents[2] / "optim" / "clmad" / "libghash_clmad.so"
_TARGET = "flock_ghash_mul"
_registered = False


def available() -> bool:
    return _SO.exists()


def _ensure_registered():
    global _registered
    if not _registered:
        lib = ctypes.cdll.LoadLibrary(str(_SO))
        jax.ffi.register_ffi_target(_TARGET, jax.ffi.pycapsule(lib.GhashMul), platform="CUDA")
        _registered = True


def mul(a, b):
    """Elementwise GF(2^128) multiply in flock's GHASH basis, via clmad.

    a, b: uint64 [..., 2] (lo, hi), broadcastable on the leading dims. Returns the
    broadcast shape. Byte-identical to `flock_zorch.field.mul`.
    """
    _ensure_registered()
    shape = jnp.broadcast_shapes(a.shape, b.shape)
    a2 = jnp.broadcast_to(a, shape).reshape(-1, 2)
    b2 = jnp.broadcast_to(b, shape).reshape(-1, 2)
    out = jax.ffi.ffi_call(_TARGET, jax.ShapeDtypeStruct(a2.shape, a2.dtype))(a2, b2)
    return out.reshape(shape)
