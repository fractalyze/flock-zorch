"""flock's zerocheck `prove_packed` — the first FULL PIOP sub-protocol with a
serializable proof — authored as a host round loop, byte-identical to flock-core's
`zerocheck::prove_packed_padded_inner`.

Proves `a(y)·b(y) ⊕ c(y) = 0 ∀ y ∈ {0,1}^m`. Structure: one univariate-skip
round-1 (URM, `gf8.round1_naive`) over K_SKIP=6 skip variables, then a multilinear
sumcheck over the remaining `m − K_SKIP` variables (the iter-10 `sumcheck`
primitives). Fiat-Shamir is the host SHA-256 `Challenger`; the bulk field arith
(`round_pair`/`fold_pair`) runs through `field.mul` (→ clmad on GPU).

The protocol fixes the inner 7 of the `r` challenge coordinates to constants
(`small`/`medium`), and the C track is pinned at round 1 (extract_c), so only AB
participate in the multilinear rounds — `final_c_eval` is an interpolation of
`round1_c` at the URM fold-point `z`. Requires `jax_enable_x64` and `zorch` on
PYTHONPATH.
"""
from __future__ import annotations

import functools

import numpy as np
import jax
import jax.numpy as jnp

from flock_zorch import field, gf8, sumcheck
from flock_zorch import _hostfield as hf
from flock_zorch.challenger import Challenger

K_SKIP = 6
N_INNER = 7  # 3 small + 4 medium fixed-constant inner dims
LABEL = b"flock-zerocheck-v0"

_MASK64 = (1 << 64) - 1


def _to_int(arr) -> int:
    a = np.asarray(arr, dtype=np.uint64)
    return int(a[0]) | (int(a[1]) << 64)


def _to_lohi(x: int) -> np.ndarray:
    return np.array([x & _MASK64, (x >> 64) & _MASK64], dtype=np.uint64)


def _phi_int(v: int) -> int:
    return _to_int(gf8.PHI_8_TABLE[v])


def small_challenges() -> list[int]:
    """[φ₈(0xF7), φ₈(0x53), φ₈(0xB5)] (flock `small_challenges_ghash`)."""
    return [_phi_int(0xF7), _phi_int(0x53), _phi_int(0xB5)]


def medium_challenges() -> list[int]:
    """[γ^E·(1+γ^E)⁻¹ for E∈{1,2,4,8}], γ^E = single bit at lo position E
    (flock `medium_challenges_ghash`)."""
    out = []
    for e in (1, 2, 4, 8):
        ge = 1 << e
        out.append(hf.mul(ge, hf.inv(1 ^ ge)))
    return out


def _prod_axis1(mat):
    """F128 product over axis 1 of [n, k, 2] via log2(k) pairwise mul steps."""
    n = mat.shape[1]
    while n > 1:
        h = n // 2
        prod = field.mul(mat[:, :h, :], mat[:, h:2 * h, :])
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
    return field.mul(num, inv_den)


@jax.jit
def _batch_inv(a):
    """Batched GF(2^128) inverse a^(2^128-2) = Π_{k=1}^{127} a^(2^k), via 127
    square-and-multiply steps (vectorized; replaces 64 host-Python Fermat invs)."""
    sq = a
    result = jnp.broadcast_to(jnp.asarray(_ONE), a.shape)
    for _ in range(127):
        sq = field.mul(sq, sq)
        result = field.mul(result, sq)
    return result


def _lagrange_weights(k_skip: int, z: int, offset: int) -> list[int]:
    """L_i(z) over the φ₈-embedded nodes PHI_8_TABLE[offset+i], i∈[0, 2^k_skip).
    offset=0 → the S domain; offset=2^k_skip → the Λ domain.

    Vectorized + jitted (the scalar O(ell²) host-Python F128 double-loop was a
    fixed ~590ms — the zerocheck's dominant cost; jit is essential — eager
    field.mul dispatches its 64-step fori per element). Same field math →
    byte-identical weights (gated)."""
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


@functools.partial(jax.jit, static_argnums=(1, 2))
def _fold_at_z_dev(rows, m: int, k_skip: int, w):
    """a_mlv[x_rest] = Σ_s witness[x_rest·ell + s]·L_s(z) (flock `fold_at_z_naive`),
    on device. rows: uint8 [2^(m-k_skip), ell]; w: uint64 [ell, 2] -> [n_chunks, 2].

    Select-and-XOR-reduce: the `[n_chunks, ell, 2]` intermediate (≈1 GB at m=26) is
    fused on the GPU instead of materialized in host numpy (the old path's bottleneck)."""
    masked = rows[:, :, None].astype(jnp.uint64) * w[None, :, :]  # 0 or w[s]
    return sumcheck._xor_reduce(masked, axis=1)


def _fold_at_z(bits, m: int, k_skip: int, weights: list[int]) -> np.ndarray:
    ell = 1 << k_skip
    n_chunks = 1 << (m - k_skip)
    rows = jnp.asarray(np.asarray(bits, np.uint8).reshape(n_chunks, ell))
    w = jnp.asarray(np.stack([_to_lohi(x) for x in weights]))  # [ell, 2]
    return _fold_at_z_dev(rows, m, k_skip, w)


_ONE = np.array([1, 0], dtype=np.uint64)

# Module-level jit cache for the per-round field ops, keyed by the `mul` callable.
# Defining these ONCE (not per prove_packed call) lets jax reuse compiled kernels
# across proofs and across rounds of the same shape — otherwise every call makes
# fresh lambdas and recompiles all n_mlv round kernels from scratch.
_JIT_CACHE: dict = {}


def _jit_round_fold(mul):
    fns = _JIT_CACHE.get(mul)
    if fns is None:
        fns = (jax.jit(lambda a, b, rr: sumcheck.round_pair(a, b, rr, mul=mul)),
               jax.jit(lambda a, b, rr: sumcheck.fold_pair(a, b, rr, mul=mul)))
        _JIT_CACHE[mul] = fns
    return fns


def prove_packed(a_bits, b_bits, c_bits, m: int, domain: bytes, mul=field.mul) -> dict:
    """Returns the ZerocheckProof fields + the claim's z / mlv_challenges / r_rest
    (the latter for the oracle's localization cross-checks)."""
    k_skip, n_mlv = K_SKIP, m - K_SKIP
    assert m >= k_skip + N_INNER, f"m must be >= {k_skip + N_INNER}"

    ch = Challenger(domain)
    ch.observe_label(LABEL)
    r_skip = ch.sample_f128_vec(k_skip)               # [6, 2]
    r_outer = ch.sample_f128_vec(m - k_skip - N_INNER)  # [m-13, 2]

    # ---- build r: r_skip ++ small ++ medium ++ r_outer ----
    r = np.zeros((m, 2), dtype=np.uint64)
    r[:k_skip] = r_skip
    for i, v in enumerate(small_challenges()):
        r[k_skip + i] = _to_lohi(v)
    for i, v in enumerate(medium_challenges()):
        r[k_skip + 3 + i] = _to_lohi(v)
    if m - k_skip - N_INNER > 0:
        r[k_skip + N_INNER:] = r_outer

    # ---- round 1 URM (== wire round1_ab/round1_c) ----
    round1_ab, round1_c = gf8.round1_naive(a_bits, b_bits, c_bits, m, k_skip, r, mul=mul)
    ch.observe_f128_slice(round1_ab)
    ch.observe_f128_slice(round1_c)
    z = ch.sample_f128()
    z_int = _to_int(z)

    # ---- c-claim: interpolate round1_c at z ----
    round1_c_int = [_to_int(round1_c[i]) for i in range(round1_c.shape[0])]
    final_c_eval = _to_lohi(_interpolate_at_z_on_lambda(round1_c_int, k_skip, z_int))

    # Per-round field ops are jitted (values identical → byte-match preserved) so
    # each round runs as ONE fused GPU kernel instead of eager op-by-op dispatch.
    # Module-level cache → kernels compile once and are reused across proofs.
    _round, _fold = _jit_round_fold(mul)

    # ---- round 2: fold witness at z + first multilinear message ----
    weights = _lagrange_weights(k_skip, z_int, 0)  # S-domain
    a_mlv = jnp.asarray(_fold_at_z(a_bits, m, k_skip, weights))
    b_mlv = jnp.asarray(_fold_at_z(b_bits, m, k_skip, weights))
    mlv_arg = np.concatenate([_ONE[None, :], r[k_skip + 1:m]], axis=0)  # [n_mlv, 2]
    msg1, msginf = _round(a_mlv, b_mlv, jnp.asarray(mlv_arg))
    rounds = [(np.asarray(msg1), np.asarray(msginf))]
    ch.observe_f128(rounds[0][0])
    ch.observe_f128(rounds[0][1])
    rhos = [ch.sample_f128()]

    # ---- rounds 3..(n_mlv+1): fold at ρ_prev, then next message ----
    for i in range(n_mlv - 1):
        r_next = np.concatenate([_ONE[None, :], r[k_skip + i + 2:m]], axis=0)
        a_mlv, b_mlv = _fold(a_mlv, b_mlv, jnp.asarray(rhos[i]))
        m1, mi = _round(a_mlv, b_mlv, jnp.asarray(r_next))
        rounds.append((np.asarray(m1), np.asarray(mi)))
        ch.observe_f128(rounds[-1][0])
        ch.observe_f128(rounds[-1][1])
        rhos.append(ch.sample_f128())

    # ---- final binding at ρ_last ----
    a_mlv, b_mlv = _fold(a_mlv, b_mlv, jnp.asarray(rhos[-1]))
    final_a_eval = np.asarray(a_mlv)[0]
    final_b_eval = np.asarray(b_mlv)[0]
    ch.observe_f128(final_a_eval)
    ch.observe_f128(final_b_eval)

    return {
        "round1_ab": round1_ab,
        "round1_c": round1_c,
        "multilinear_rounds": rounds,
        "final_a_eval": final_a_eval,
        "final_b_eval": final_b_eval,
        "final_c_eval": final_c_eval,
        # claim cross-checks:
        "z": z,
        "mlv_challenges": np.stack(rhos),
        "r_rest": r[k_skip:],
    }
