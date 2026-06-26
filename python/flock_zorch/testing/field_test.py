"""Known-answer test for flock_zorch.field GF(2^128) multiply.

Anchored on flock's `ghash_reduction_smoking_gun` vectors (gf2_128.rs:665).
Runs on whatever backend JAX_PLATFORMS selects.
"""
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flock_zorch import field  # noqa: E402

# (a=(lo,hi), b=(lo,hi), expected=(lo,hi))
_VECTORS = [
    ((2, 0), (0, 1 << 63), (0x87, 0)),  # x * x^127 = 0x87
    ((2, 0), (2, 0), (4, 0)),           # x * x = x^2
    ((0, 1), (0, 1), (0x87, 0)),        # x^64 * x^64 = 0x87
    ((2, 0), (1 << 63, 0), (0, 1)),     # x * x^63 = x^64
    ((1, 0), (2, 0), (2, 0)),           # 1 * x = x
    ((0, 0), (12345, 678), (0, 0)),     # 0 * anything = 0
]


def _mul(a_lohi, b_lohi):
    a = jnp.asarray(np.array([a_lohi], dtype=np.uint64))
    b = jnp.asarray(np.array([b_lohi], dtype=np.uint64))
    return tuple(int(v) for v in np.asarray(field.mul(a, b))[0])


def test_smoking_gun():
    for a, b, want in _VECTORS:
        got = _mul(a, b)
        assert got == want, f"{a} * {b}: got {got}, want {want}"


if __name__ == "__main__":
    test_smoking_gun()
    print(f"field KAT: PASS on {jax.default_backend()} ({len(_VECTORS)} vectors)")
