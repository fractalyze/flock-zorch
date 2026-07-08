"""flock F128 (GF(2^128), GHASH basis) serialization helpers for flock-zorch.

flock (succinctlabs/flock) represents F128 in the GHASH/POLYVAL polynomial basis
(irreducible p(x) = x^128 + x^7 + x^2 + x + 1, natural non-bit-reflected order) as
two u64 limbs {lo, hi}: `lo` holds the coefficients of x^0..x^63, `hi` of
x^64..x^127. On a little-endian host that is exactly flock's 16-byte serialization,
and exactly the storage of zk_dtypes' `binary_field_ghash` dtype (same basis,
verified 2*2 = 4). Device arithmetic therefore runs on the dtype (`*` / `+` /
`jnp.sum`); this module only bridges representations at the edges (the golden-fixture
readers and the challenger's F128 byte serde).

`to_ghash`/`from_ghash` are the device uint64-lane <-> `binary_field_ghash` bitcast.
They route through uint32 lanes: the direct `uint64[..,2] <-> ghash` bitcast silently
miscompiles on the CPU PJRT path (a fractalyze/xla BitcastConvertType bug), while
`uint32[..,4] <-> ghash` is correct on both backends and is the dtype's native lane
width. `_to_int`/`_to_lohi` are the host int <-> uint64-lane serde (bit i =
coefficient of x^i). Requires `jax_enable_x64`.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

U64 = jnp.uint64

LOG_PACKING = 7  # an F128 packs 2^7 = 128 bits; witness log-size m -> 2^(m-7) packed elems

_GHASH = jnp.binary_field_ghash


def to_ghash(a):
    """uint64 `[..., 2]` (lo, hi) F128 -> `binary_field_ghash [...]`, via uint32 lanes.

    flock stores an F128 as its little-endian bytes; the dtype is the same bytes,
    so this is a pure bitcast. It routes through uint32 lanes because the direct
    uint64<->ghash bitcast silently miscompiles on the CPU PJRT path (a
    fractalyze/xla BitcastConvertType bug), while the uint32 bitcast is correct on
    both backends and is the dtype's native lane width."""
    a = jnp.asarray(a, U64)
    u32 = jax.lax.bitcast_convert_type(a, jnp.uint32).reshape(*a.shape[:-1], 4)
    return jax.lax.bitcast_convert_type(u32, _GHASH)


def from_ghash(g):
    """`binary_field_ghash [...]` -> uint64 `[..., 2]` (lo, hi). Inverse of `to_ghash`."""
    u32 = jax.lax.bitcast_convert_type(g, jnp.uint32)
    return jax.lax.bitcast_convert_type(u32.reshape(*u32.shape[:-1], 2, 2), U64)


def from_ghash_host(g) -> np.ndarray:
    """`from_ghash` materialized to host numpy — for the host-consumed uint64 lanes
    (verify replay, transcript serde, the chain / lincheck reductions)."""
    return np.asarray(from_ghash(g))


# ---- host int <-> uint64-lane serialization (bit i = coefficient of x^i) ----
_MASK64 = (1 << 64) - 1


def _to_int(arr) -> int:
    """F128 uint64 [.., 2] (lo, hi) -> Python int (bit i = coefficient of x^i)."""
    a = np.asarray(arr, dtype=np.uint64)
    return int(a[0]) | (int(a[1]) << 64)


def _to_lohi(x: int) -> np.ndarray:
    """Python-int F128 -> uint64 [2] (lo, hi)."""
    return np.array([x & _MASK64, (x >> 64) & _MASK64], dtype=np.uint64)


# ---- host int <-> binary_field_ghash (for small sequential host precomputes:
# flock's fixed challenge constants, the c-claim interpolation, verify replay).
# The dtype is the same LE bytes as the lo/hi lanes, so this is a numpy view; the
# field arithmetic (`*`, `** -1`) is zk_dtypes' host impl — a single field source
# of truth shared with the device path. ----
_GHASH_HOST = np.dtype(_GHASH)


def _ints_to_ghash(vals) -> np.ndarray:
    """`list[int]` F128 (bit i = coeff x^i) -> host `binary_field_ghash [n]`."""
    lohi = np.array([_to_lohi(v) for v in vals], np.uint64)
    return lohi.reshape(-1).view(_GHASH_HOST)


def _int_to_ghash(v: int):
    """Python-int F128 -> host `binary_field_ghash` scalar."""
    return _ints_to_ghash([v])[0]


def _ghash_to_int(g) -> int:
    """Host `binary_field_ghash` scalar -> Python int (bit i = coeff x^i)."""
    return _to_int(np.asarray(g).reshape(1).view(np.uint64))
