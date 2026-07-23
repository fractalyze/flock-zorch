"""flock F128 (GF(2^128), GHASH basis) serialization helpers for flock-zorch.

flock (succinctlabs/flock) represents F128 in the GHASH/POLYVAL polynomial basis
(irreducible p(x) = x^128 + x^7 + x^2 + x + 1, natural non-bit-reflected order) as
two u64 limbs {lo, hi}: `lo` holds the coefficients of x^0..x^63, `hi` of
x^64..x^127. On a little-endian host that is exactly flock's 16-byte serialization,
and exactly the storage of zk_dtypes' `binary_field_ghash` dtype (same basis,
verified 2*2 = 4). Device arithmetic therefore runs on the dtype (`*` / `+` /
`fnp.sum`); this module only bridges representations at the edges (the golden-fixture
readers and the challenger's F128 byte serde).

`to_ghash`/`from_ghash` are the device uint64-lane <-> `binary_field_ghash` bitcast
(a pure reinterpret — flock's {lo, hi} limbs are the dtype's storage bytes);
`_lanes_to_ghash`/`_ghash_to_lanes` are the same reinterpret as a host numpy view.
The prove path holds native ghash end-to-end (flock-zorch#155): nothing here
forces a device->host materialization — `to_lanes` is the byte-gate readers'
compare-edge lift, and the only remaining host crossing inside a prove is the
challenger's SHA-256 byte serde. Requires `jax_enable_x64`.
"""
from __future__ import annotations

import numpy as np
import frx
import frx.numpy as fnp

U64 = fnp.uint64

LOG_PACKING = 7  # an F128 packs 2^7 = 128 bits; witness log-size m -> 2^(m-7) packed elems

_GHASH = fnp.binary_field_ghash


def to_ghash(a):
    """uint64 `[..., 2]` (lo, hi) F128 -> `binary_field_ghash [...]`.

    flock stores an F128 as its little-endian bytes and the dtype is the same
    bytes, so this is a pure bitcast."""
    return frx.lax.bitcast_convert_type(fnp.asarray(a, U64), _GHASH)


def from_ghash(g):
    """`binary_field_ghash [...]` -> uint64 `[..., 2]` (lo, hi). Inverse of `to_ghash`."""
    return frx.lax.bitcast_convert_type(g, U64)


def zeros(n: int):
    """`binary_field_ghash [n]` of field zeros — a bitcast of zero bytes, NOT
    `fnp.zeros(n, binary_field_ghash)` (an int->ghash convert is unimplemented; a
    scalar default even emits an S64->ghash convert at compile — see CLAUDE.md)."""
    return frx.lax.bitcast_convert_type(fnp.zeros((n, 2), U64), _GHASH)


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


