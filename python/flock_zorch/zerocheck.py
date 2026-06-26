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

import numpy as np
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


def _lagrange_weights(k_skip: int, z: int, offset: int) -> list[int]:
    """L_i(z) over the φ₈-embedded nodes PHI_8_TABLE[offset+i], i∈[0, 2^k_skip).
    offset=0 → the S domain; offset=2^k_skip → the Λ domain."""
    ell = 1 << k_skip
    nodes = [_phi_int(offset + i) for i in range(ell)]
    weights = []
    for i in range(ell):
        si = nodes[i]
        num, den = 1, 1
        for j in range(ell):
            if j == i:
                continue
            sj = nodes[j]
            num = hf.mul(num, z ^ sj)
            den = hf.mul(den, si ^ sj)
        weights.append(hf.mul(num, hf.inv(den)))
    return weights


def _interpolate_at_z_on_lambda(values_int: list[int], k_skip: int, z: int) -> int:
    """Σ_i L_i^Λ(z)·values[i] (flock `interpolate_at_z_on_lambda`)."""
    w = _lagrange_weights(k_skip, z, 1 << k_skip)
    acc = 0
    for i in range(1 << k_skip):
        acc ^= hf.mul(w[i], values_int[i])
    return acc


def _fold_at_z(bits, m: int, k_skip: int, weights: list[int]) -> np.ndarray:
    """a_mlv[x_rest] = Σ_s witness[x_rest·ell + s]·L_s(z) (flock `fold_at_z_naive`).
    Returns uint64 [2^(m-k_skip), 2]."""
    ell = 1 << k_skip
    n_chunks = 1 << (m - k_skip)
    rows = np.asarray(bits, np.uint8).reshape(n_chunks, ell)
    w = np.stack([_to_lohi(x) for x in weights])  # [ell, 2]
    masked = w[None, :, :] * rows[:, :, None]      # [n_chunks, ell, 2], 0 or w[s]
    return np.bitwise_xor.reduce(masked, axis=1).astype(np.uint64)


_ONE = np.array([1, 0], dtype=np.uint64)


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

    # ---- round 2: fold witness at z + first multilinear message ----
    weights = _lagrange_weights(k_skip, z_int, 0)  # S-domain
    a_mlv = jnp.asarray(_fold_at_z(a_bits, m, k_skip, weights))
    b_mlv = jnp.asarray(_fold_at_z(b_bits, m, k_skip, weights))
    mlv_arg = np.concatenate([_ONE[None, :], r[k_skip + 1:m]], axis=0)  # [n_mlv, 2]
    msg1, msginf = sumcheck.round_pair(a_mlv, b_mlv, jnp.asarray(mlv_arg), mul=mul)
    rounds = [(np.asarray(msg1), np.asarray(msginf))]
    ch.observe_f128(rounds[0][0])
    ch.observe_f128(rounds[0][1])
    rhos = [ch.sample_f128()]

    # ---- rounds 3..(n_mlv+1): fold at ρ_prev, then next message ----
    for i in range(n_mlv - 1):
        r_next = np.concatenate([_ONE[None, :], r[k_skip + i + 2:m]], axis=0)
        a_mlv, b_mlv = sumcheck.fold_pair(a_mlv, b_mlv, jnp.asarray(rhos[i]), mul=mul)
        m1, mi = sumcheck.round_pair(a_mlv, b_mlv, jnp.asarray(r_next), mul=mul)
        rounds.append((np.asarray(m1), np.asarray(mi)))
        ch.observe_f128(rounds[-1][0])
        ch.observe_f128(rounds[-1][1])
        rhos.append(ch.sample_f128())

    # ---- final binding at ρ_last ----
    a_mlv, b_mlv = sumcheck.fold_pair(a_mlv, b_mlv, jnp.asarray(rhos[-1]), mul=mul)
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
