"""Device precompute for zerocheck's round-1 c-claim and round-2 binding — the
Lagrange-weight / batched-inverse / fold-at-z cluster, kept out of zerocheck.py so
prove_packed reads as the PIOP. Byte-identical to flock's zerocheck/{univariate_skip,
multilinear}. The deeply-sequential Lagrange/Fermat work routes through _LMUL (clmad
when available) — one hardware carryless-mul per step instead of the software fori.

Requires jax_enable_x64.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from flock_zorch import field, gf8
from flock_zorch import _hostfield as hf
from flock_zorch.field import _to_int, _to_lohi

_ONE = np.array([1, 0], dtype=np.uint64)

# The small fixed-size Lagrange/inverse helpers here are deeply sequential (254 muls
# for the Fermat inverse). They use the software `field.mul`.
_LMUL = field.mul


def _phi_int(v: int) -> int:
    return _to_int(gf8.PHI_8_TABLE[v])


def _prod_axis1(mat):
    """F128 product over axis 1 of [n, k, 2] via log2(k) pairwise mul steps."""
    n = mat.shape[1]
    while n > 1:
        h = n // 2
        prod = _LMUL(mat[:, :h, :], mat[:, h:2 * h, :])
        if n % 2:
            prod = jnp.concatenate([prod, mat[:, 2 * h:, :]], axis=1)
        mat = prod
        n = mat.shape[1]
    return mat[:, 0, :]


@jax.jit
def _lag_numden(s, zf):
    """num[i]=Π_{j≠i}(z+s_j), den[i]=Π_{j≠i}(s_i+s_j); diagonal terms set to 1."""
    ell = s.shape[0]
    eye = jnp.eye(ell, dtype=bool)[:, :, None]
    one = jnp.asarray(_ONE)
    num_mat = jnp.where(eye, one, jnp.broadcast_to((zf ^ s)[None, :, :], (ell, ell, 2)))
    den_mat = jnp.where(eye, one, s[:, None, :] ^ s[None, :, :])
    return _prod_axis1(num_mat), _prod_axis1(den_mat)


@jax.jit
def _lag_w(num, inv_den):
    return _LMUL(num, inv_den)


@jax.jit
def _batch_inv(a):
    """Batched GF(2^128) inverse a^(2^128-2) = Π_{k=1}^{127} a^(2^k), via 127
    square-and-multiply steps (vectorized; replaces 64 host-Python Fermat invs)."""
    sq = a
    result = jnp.broadcast_to(jnp.asarray(_ONE), a.shape)
    for _ in range(127):
        sq = _LMUL(sq, sq)
        result = _LMUL(result, sq)
    return result


def _lagrange_weights(k_skip: int, z: int, offset: int) -> list[int]:
    """L_i(z) over the φ₈-embedded nodes PHI_8_TABLE[offset+i], i∈[0, 2^k_skip).
    offset=0 → the S domain; offset=2^k_skip → the Λ domain.

    Vectorized + jitted (replaces the scalar O(ell²) host-Python F128 double-loop;
    jit is essential — an eager field.mul dispatches its 64-step fori per element).
    Same field math → byte-identical weights (gated)."""
    ell = 1 << k_skip
    s = jnp.asarray(np.stack([_to_lohi(_phi_int(offset + i)) for i in range(ell)]))  # [ell,2]
    num, den = _lag_numden(s, jnp.asarray(_to_lohi(z)))
    return [_to_int(x) for x in np.asarray(_lag_w(num, _batch_inv(den)))]


def _interpolate_at_z_on_lambda(values_int: list[int], k_skip: int, z: int) -> int:
    """Σ_i L_i^Λ(z)·values[i] (flock `interpolate_at_z_on_lambda`)."""
    w = _lagrange_weights(k_skip, z, 1 << k_skip)
    acc = 0
    for i in range(1 << k_skip):
        acc ^= hf.mul(w[i], values_int[i])
    return acc


@jax.jit
def _fold_at_z_dev(rows, w):
    """a_mlv[x_rest] = Σ_s witness[x_rest·ell + s]·L_s(z) (flock `fold_at_z_naive`),
    on device. rows: uint8 [2^(m-k_skip), ell]; w: uint64 [ell, 2] -> [n_chunks, 2].

    Select-and-XOR-reduce: the large `[n_chunks, ell, 2]` intermediate is fused on
    the GPU instead of materialized in host numpy."""
    masked = rows[:, :, None].astype(jnp.uint64) * w[None, :, :]  # 0 or w[s]
    return field.sum(masked, axis=1)


def _fold_at_z_rows(rows, weights: list[int]) -> np.ndarray:
    """fold_at_z from device witness rows (uint8 [2^(m-k_skip), 2^k_skip]) — so the
    witness transferred for round1 is reused here instead of re-sent."""
    w = jnp.asarray(np.stack([_to_lohi(x) for x in weights]))  # [ell, 2]
    return _fold_at_z_dev(rows, w)
