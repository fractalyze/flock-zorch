"""Byte-match gate for the compiler-native AES-basis GF(2⁸) NTT.

`lax.ntt` over `binary_field_gf8_aes` dispatches the compiler's field-generic
LCH14 additive NTT; flock's zerocheck round-1 S→Λ extension re-expresses as
base-subspace transforms only (inverse NTT size ℓ → zero-pad coefficients to
2ℓ → forward NTT size 2ℓ → second half = the β=ℓ coset), so no coset-offset
kernel support is needed. Checks, each byte-exact (the generic lax.ntt
contract — roundtrip + CPU/GPU byte parity — is jax's own test suite's job):

  1. The re-expressed extension == the hand-rolled coset-twiddle device path
     (`_gf8_device`), on random rows.
  2. Round-1 with the re-expressed extension byte-matches the `gf8_urm`
     golden (`round1_ab` / `round1_c`) — anchoring the dtype to unmodified
     flock.

Run (needs jax_enable_x64 + a jax stack with binary_field_gf8_aes):
    PYTHONPATH=python:../zorch <venv>/bin/python \
        python/flock_zorch/field/testing/gf8_aes_ntt_gate.py
"""
import functools
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import zk_dtypes  # noqa: E402
from jax import lax  # noqa: E402

from flock_zorch import field, sumcheck  # noqa: E402
from flock_zorch.field import gf8, _gf8_device  # noqa: E402
from flock_zorch.field.testing import gf8_urm_oracle_test  # noqa: E402

_AES = np.dtype(zk_dtypes.binary_field_gf8_aes)


def _to_aes(x):
    return lax.bitcast_convert_type(x, _AES)


def _to_u8(x):
    return lax.bitcast_convert_type(x, jnp.uint8)


def _extend_lax(rows, k_skip: int):
    """S→Λ extension via base-subspace `lax.ntt` only: INTT_ℓ → zero-pad to
    2ℓ → NTT_2ℓ → second half (the β=ℓ coset). rows: uint8 [n, ℓ]."""
    ell = 1 << k_skip
    coeffs = lax.ntt(_to_aes(rows), ntt_type="INTT", ntt_length=ell)
    padded = jnp.concatenate([coeffs, jnp.zeros_like(coeffs)], axis=-1)
    evals = lax.ntt(padded, ntt_type="NTT", ntt_length=2 * ell)
    return _to_u8(evals[..., ell:])


def _check_extension_matches_coset_path():
    rng = np.random.default_rng(7)
    for k_skip in (3, 6):
        ell = 1 << k_skip
        rows = rng.integers(0, 256, size=(32, ell), dtype=np.uint8)
        tw_s = jnp.asarray(gf8._compute_twiddles(k_skip, 0))
        tw_l = jnp.asarray(gf8._compute_twiddles(k_skip, ell))
        legacy = _gf8_device._fft_dev(
            _gf8_device._ifft_dev(jnp.asarray(rows), tw_s, k_skip), tw_l, k_skip)
        got = _extend_lax(jnp.asarray(rows), k_skip)
        assert np.array_equal(np.asarray(got), np.asarray(legacy)), \
            f"re-expressed extension != coset path, k_skip={k_skip}"


@functools.partial(jax.jit, static_argnums=(3,))
def _round1_core_lax(a, b, c, k_skip, eqx):
    """`_gf8_device._round1_core` with the S→Λ extension re-expressed through
    `lax.ntt` (test-only; the production swap is tracked separately)."""
    a_l = _to_aes(_extend_lax(a, k_skip))
    b_l = _to_aes(_extend_lax(b, k_skip))
    c_l = _extend_lax(c, k_skip)
    ab = _to_u8(a_l * b_l).astype(jnp.int32)
    phi = jnp.asarray(gf8.PHI_8_TABLE)
    phi_ab = field.to_ghash(phi[ab])
    phi_c = field.to_ghash(phi[c_l.astype(jnp.int32)])
    eqx_g = field.to_ghash(eqx)
    return (field.from_ghash(jnp.sum(eqx_g * phi_ab, axis=0)),
            field.from_ghash(jnp.sum(eqx_g * phi_c, axis=0)))


def _check_round1_matches_golden(path: Path | None = None):
    path = path or (gf8_urm_oracle_test._artifacts_dir() / "gf8_urm_golden.bin")
    raw = path.read_bytes()
    assert raw[:8] == gf8_urm_oracle_test._MAGIC, f"bad magic {raw[:8]!r}"
    rd = gf8_urm_oracle_test._Reader(raw)
    rd.off = 8
    rd.f128(256)  # phi8 table — checked by the URM oracle gate
    configs = []
    for _ in range(rd.u64()):
        m, k_skip = rd.u64(), rd.u64()
        n = 1 << m
        a, b, c = rd.bits(n), rd.bits(n), rd.bits(n)
        r = rd.f128(m)
        ab_golden = rd.f128(1 << k_skip)
        c_golden = rd.f128(1 << k_skip)
        a_r = gf8.witness_to_rows(a, m, k_skip)
        b_r = gf8.witness_to_rows(b, m, k_skip)
        c_r = gf8.witness_to_rows(c, m, k_skip)
        eqx = sumcheck.build_eq_fused(jnp.asarray(r[k_skip:]))[:, None, :]
        p_ab, p_c = _round1_core_lax(a_r, b_r, c_r, k_skip, eqx)
        assert np.array_equal(np.asarray(p_ab), ab_golden), \
            f"round1_ab mismatch via lax.ntt extension (m={m},k={k_skip})"
        assert np.array_equal(np.asarray(p_c), c_golden), \
            f"round1_c mismatch via lax.ntt extension (m={m},k={k_skip})"
        configs.append((m, k_skip))
    return configs


def test_gf8_aes_ntt_gate():
    _check_extension_matches_coset_path()
    _check_round1_matches_golden()


if __name__ == "__main__":
    _check_extension_matches_coset_path()
    print("re-expressed S→Λ extension == coset path: PASS")
    cfgs = _check_round1_matches_golden()
    print(f"round1 via lax.ntt extension byte-match vs flock golden: PASS {cfgs}")
