"""Byte-match gate for the GF(2^128) multiply (port target #1).

Reads golden (a, b, a*b) triples produced by the flock-core reference
(`examples/dump_field_mul.rs` -> `software::ghash_mul`) and asserts the jax port
in `flock_zorch.field.mul` reproduces every product byte-for-byte.

Run directly (backend chosen by JAX_PLATFORMS):
    JAX_PLATFORMS=cuda PYTHONPATH=python <venv>/bin/python \
        python/flock_zorch/testing/field_oracle_test.py
"""
import os
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flock_zorch import field  # noqa: E402

_MAGIC = b"FLKMUL01"


def _artifacts_dir() -> Path:
    env = os.environ.get("FLOCK_ZORCH_ARTIFACTS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "artifacts"


def _load_golden(path: Path):
    raw = path.read_bytes()
    assert raw[:8] == _MAGIC, f"bad magic {raw[:8]!r} in {path}"
    n = int.from_bytes(raw[8:16], "little")
    off, blk = 16, n * 16
    a = np.frombuffer(raw, dtype=np.uint64, count=n * 2, offset=off).reshape(n, 2)
    b = np.frombuffer(raw, dtype=np.uint64, count=n * 2, offset=off + blk).reshape(n, 2)
    out = np.frombuffer(raw, dtype=np.uint64, count=n * 2, offset=off + 2 * blk).reshape(n, 2)
    return n, a, b, out


def run(path: Path | None = None) -> int:
    path = path or (_artifacts_dir() / "field_mul_golden.bin")
    n, a, b, golden = _load_golden(path)
    got = np.asarray(jax.jit(field.mul)(jnp.asarray(a), jnp.asarray(b)))
    if not np.array_equal(got, golden):
        diff = np.flatnonzero(np.any(got != golden, axis=1))
        i = int(diff[0])
        raise AssertionError(
            f"mismatch at {len(diff)}/{n} rows; first @ {i}: "
            f"a={a[i].tolist()} b={b[i].tolist()} "
            f"got={got[i].tolist()} golden={golden[i].tolist()}"
        )
    return n


def test_field_mul_oracle():
    run()


if __name__ == "__main__":
    count = run()
    print(f"field-mul byte-match: PASS on {jax.default_backend()} ({count} pairs)")
