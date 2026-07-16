"""flock F128 (GF(2^128), GHASH basis) serialization helpers for flock-zorch.

flock (succinctlabs/flock) represents F128 in the GHASH/POLYVAL polynomial basis
(irreducible p(x) = x^128 + x^7 + x^2 + x + 1, natural non-bit-reflected order) as
two u64 limbs {lo, hi}: `lo` holds the coefficients of x^0..x^63, `hi` of
x^64..x^127. On a little-endian host that is exactly flock's 16-byte serialization,
and exactly the storage of zk_dtypes' `binary_field_ghash` dtype (same basis,
verified 2*2 = 4). Device arithmetic therefore runs on the dtype (`*` / `+` /
`jnp.sum`); this module only bridges representations at the edges (the golden-fixture
readers and the challenger's F128 byte serde).

`to_ghash`/`from_ghash` are the device uint64-lane <-> `binary_field_ghash` bitcast
(a pure reinterpret — flock's {lo, hi} limbs are the dtype's storage bytes);
`_lanes_to_ghash`/`_ghash_to_lanes` are the same reinterpret as a host numpy view.
`_to_int`/`_to_lohi` are the host int <-> uint64-lane serde (bit i = coefficient
of x^i). Requires `jax_enable_x64`.
"""
from __future__ import annotations

import numpy as np
import frx
import frx.numpy as jnp

U64 = jnp.uint64

LOG_PACKING = 7  # an F128 packs 2^7 = 128 bits; witness log-size m -> 2^(m-7) packed elems

_GHASH = jnp.binary_field_ghash


def to_ghash(a):
    """uint64 `[..., 2]` (lo, hi) F128 -> `binary_field_ghash [...]`.

    flock stores an F128 as its little-endian bytes and the dtype is the same
    bytes, so this is a pure bitcast."""
    return frx.lax.bitcast_convert_type(jnp.asarray(a, U64), _GHASH)


def from_ghash(g):
    """`binary_field_ghash [...]` -> uint64 `[..., 2]` (lo, hi). Inverse of `to_ghash`."""
    return frx.lax.bitcast_convert_type(g, U64)


def from_ghash_host(g) -> np.ndarray:
    """`from_ghash` materialized to host numpy — for the host-consumed uint64 lanes
    (verify replay, transcript serde, the chain / lincheck reductions)."""
    return np.asarray(from_ghash(g))


def to_lanes(x) -> np.ndarray:
    """Any F128 — native `binary_field_ghash` OR uint64 `[..., 2]` lanes — to host
    uint64 `[..., 2]`. The proof holds ghash; byte-gate readers pass `got` through
    this so a ghash array and its lane serialization compare equal."""
    a = np.asarray(x)
    if a.dtype == _GHASH_HOST:
        return a.reshape(-1).view(np.uint64).reshape(*a.shape, 2)
    return a.astype(np.uint64)


# ---- host uint64-lane <-> binary_field_ghash (for small sequential host
# precomputes: flock's fixed challenge constants, verify replay). The dtype is the
# same LE bytes as the lo/hi lanes, so this is a numpy view — no device round-trip;
# the field arithmetic (`*`, `** -1`) is zk_dtypes' host impl, a single field source
# of truth shared with the device path. ----
_GHASH_HOST = np.dtype(_GHASH)


def _lanes_to_ghash(lanes) -> np.ndarray:
    """Host `uint64 [..., 2]` (lo, hi) -> `binary_field_ghash [...]`."""
    a = np.ascontiguousarray(lanes, np.uint64)
    return a.reshape(-1).view(_GHASH_HOST).reshape(a.shape[:-1])


def _ghash_to_lanes(g) -> np.ndarray:
    """Host `binary_field_ghash [...]` -> `uint64 [..., 2]`. Inverse of `_lanes_to_ghash`."""
    g = np.asarray(g)
    return g.reshape(-1).view(np.uint64).reshape(*g.shape, 2)


# ---- host int <-> uint64-lane serialization (bit i = coefficient of x^i). Sole
# consumer is the basefold verify oracle's independent big-int GF(2^128) multiply,
# which must NOT share the dtype impl it gates. ----
_MASK64 = (1 << 64) - 1


def _to_int(arr) -> int:
    """F128 uint64 [.., 2] (lo, hi) -> Python int (bit i = coefficient of x^i)."""
    a = np.asarray(arr, dtype=np.uint64)
    return int(a[0]) | (int(a[1]) << 64)


def _to_lohi(x: int) -> np.ndarray:
    """Python-int F128 -> uint64 [2] (lo, hi)."""
    return np.array([x & _MASK64, (x >> 64) & _MASK64], dtype=np.uint64)
