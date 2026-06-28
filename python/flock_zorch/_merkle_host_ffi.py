"""Host SHA-NI Merkle path (the "SHA-256 Merkle off-GPU" optimization), kept out of
merkle.py so the GPU construction reads cleanly. GPU SHA-256 (the 64-round fori ×
log(n) sequential levels) is the prover's Merkle wall and loses to CPU SHA-NI.
`optim/merkle_ffi/merkle_ffi.rs`, built into `target/release/libflock_zorch.so`,
re-exports flock's own rayon+SHA-NI `merkle::merkle_tree` over a C ABI. Byte-identical
to the GPU path (same flock construction), so the gates still pin it; opt in via
`use_host_sha=True`. Lives in the package dir so `__file__` resolves the cdylib at
../../target/release/.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "target", "release", "libflock_zorch.so"
)
_lib = None


def _host_lib():
    """Lazily dlopen the flock-zorch cdylib and bind the Merkle FFI symbols."""
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH):
            raise RuntimeError(
                f"host SHA Merkle requested but {_LIB_PATH} is missing — "
                "build it with `cargo build --release`."
            )
        lib = ctypes.CDLL(_LIB_PATH)
        # (data, data_len, n_leaves, out)
        sig = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
        lib.flock_merkle_tree.argtypes = sig
        lib.flock_merkle_root.argtypes = sig
        _lib = lib
    return _lib


def host_sha_available() -> bool:
    """True if the cdylib exists (so `use_host_sha=True` will work)."""
    return os.path.exists(_LIB_PATH)


def _merkle_tree_host(leaves) -> np.ndarray:
    """Full flat tree via flock's rayon+SHA-NI `merkle_tree`. leaves uint8
    [n, leaf_size] -> uint8 [2n-1, 32], byte-identical to the GPU path."""
    leaves = np.ascontiguousarray(leaves, dtype=np.uint8)
    n_leaves = int(leaves.shape[0])
    data = leaves.ravel()
    out = np.empty((2 * n_leaves - 1, 32), dtype=np.uint8)
    _host_lib().flock_merkle_tree(data.ctypes.data, data.size, n_leaves, out.ctypes.data)
    return out


def _merkle_root_host(leaves) -> np.ndarray:
    """32-byte root via flock's rayon+SHA-NI `merkle_tree` (root only)."""
    leaves = np.ascontiguousarray(leaves, dtype=np.uint8)
    n_leaves = int(leaves.shape[0])
    data = leaves.ravel()
    out = np.empty(32, dtype=np.uint8)
    _host_lib().flock_merkle_root(data.ctypes.data, data.size, n_leaves, out.ctypes.data)
    return out
