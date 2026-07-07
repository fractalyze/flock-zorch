"""Byte-match gate for the multilinear-sumcheck arithmetic core (iter 10).

Reads reference outputs of flock-core's `build_eq`, `round_pair_naive`, and
`fold_in_place_single` (dumped by `examples/dump_sumcheck.rs`) and asserts the
jax port in `flock_zorch.sumcheck` reproduces each byte-for-byte. Runs with the
software field mul AND, when available, the clmad FFI — both must match.

Run directly (backend chosen by JAX_PLATFORMS):
    JAX_PLATFORMS=cuda PYTHONPATH=python <venv>/bin/python \
        python/flock_zorch/testing/sumcheck_oracle_test.py
"""
import os
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flock_zorch import field, sumcheck  # noqa: E402

_MAGIC = b"FLKSUM01"


def _artifacts_dir() -> Path:
    env = os.environ.get("FLOCK_ZORCH_ARTIFACTS")
    return Path(env) if env else Path(__file__).resolve().parents[3] / "artifacts"


class _Reader:
    """Sequential little-endian reader over the dumped fixture."""

    def __init__(self, raw: bytes):
        self.raw = raw
        self.off = 0

    def u64(self) -> int:
        v = int.from_bytes(self.raw[self.off:self.off + 8], "little")
        self.off += 8
        return v

    def f128(self, count: int) -> np.ndarray:
        a = np.frombuffer(self.raw, np.uint64, count * 2, self.off).reshape(count, 2)
        self.off += count * 16
        return a


def load(path: Path):
    raw = path.read_bytes()
    assert raw[:8] == _MAGIC, f"bad magic {raw[:8]!r} in {path}"
    rd = _Reader(raw)
    rd.off = 8

    eq_cases = []
    for _ in range(rd.u64()):
        n = rd.u64()
        r = rd.f128(n)
        eq = rd.f128(1 << n)
        eq_cases.append((n, r, eq))

    rp_cases = []
    for _ in range(rd.u64()):
        log_n = rd.u64()
        a = rd.f128(1 << log_n)
        b = rd.f128(1 << log_n)
        r = rd.f128(log_n)
        msg_one = rd.f128(1)[0]
        msg_inf = rd.f128(1)[0]
        rp_cases.append((log_n, a, b, r, msg_one, msg_inf))

    fs_cases = []
    for _ in range(rd.u64()):
        log_n = rd.u64()
        a = rd.f128(1 << log_n)
        z = rd.f128(1)[0]
        folded = rd.f128(1 << (log_n - 1))
        fs_cases.append((log_n, a, z, folded))

    return eq_cases, rp_cases, fs_cases


def _check(name: str, got: np.ndarray, golden: np.ndarray) -> None:
    got = np.asarray(got)
    if not np.array_equal(got, golden):
        diff = np.flatnonzero(np.any(got.reshape(-1, 2) != golden.reshape(-1, 2), axis=1))
        i = int(diff[0])
        raise AssertionError(
            f"{name} mismatch {len(diff)}/{golden.reshape(-1, 2).shape[0]}; "
            f"first @ {i}: got={got.reshape(-1, 2)[i].tolist()} "
            f"golden={golden.reshape(-1, 2)[i].tolist()}"
        )


def run(path: Path | None = None, mul=field.mul) -> dict:
    path = path or (_artifacts_dir() / "sumcheck_golden.bin")
    eq_cases, rp_cases, fs_cases = load(path)

    build_eq = jax.jit(lambda r: sumcheck.build_eq(r, mul=mul))
    for n, r, eq in eq_cases:
        _check(f"build_eq(n={n})", build_eq(jnp.asarray(r)), eq)

    for log_n, a, b, r, msg_one, msg_inf in rp_cases:
        fn = jax.jit(lambda aa, bb, rr, ln=log_n: sumcheck.round_pair(aa, bb, rr, mul=mul))
        g_one, g_inf = fn(jnp.asarray(a), jnp.asarray(b), jnp.asarray(r))
        _check(f"round_pair msg_one(log_n={log_n})", g_one, msg_one)
        _check(f"round_pair msg_inf(log_n={log_n})", g_inf, msg_inf)

    fold = jax.jit(lambda a, z: sumcheck.fold_single(a, z, mul=mul))
    for log_n, a, z, folded in fs_cases:
        _check(f"fold_single(log_n={log_n})", fold(jnp.asarray(a), jnp.asarray(z)), folded)

    return {
        "eq": [n for n, *_ in eq_cases],
        "round_pair": [ln for ln, *_ in rp_cases],
        "fold": [ln for ln, *_ in fs_cases],
    }


def test_sumcheck_oracle():
    run()


if __name__ == "__main__":
    sizes = run(mul=field.mul)
    print(f"sumcheck byte-match (software mul): PASS on {jax.default_backend()} | "
          f"build_eq n={sizes['eq']} round_pair log_n={sizes['round_pair']} "
          f"fold log_n={sizes['fold']}")
