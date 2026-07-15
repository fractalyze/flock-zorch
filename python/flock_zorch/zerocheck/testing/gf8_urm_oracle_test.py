"""Byte-match gate for φ₈ + the zerocheck round-1 URM (`flock_zorch.zerocheck._urm`).

Two checks against flock-core (`examples/dump_gf8_urm.rs`):
  1. φ₈ — the F2-linear table built from 8 basis rows equals flock's PHI_8_TABLE
     (256 entries) byte-for-byte. Localizes any φ₈ transcription error.
  2. round1_naive — the URM message (round1_ab, round1_c) matches flock's
     reference for each (m, k_skip), exercising the F8 inverse/forward NTT, the
     φ₈ lift, and the F128 eq-accumulation end to end.

Run (needs jax_enable_x64; host F8 + device F128):
    PYTHONPATH=python:../zorch <venv>/bin/python \
        python/flock_zorch/zerocheck/testing/gf8_urm_oracle_test.py
"""
import os
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from flock_zorch import field  # noqa: E402
from flock_zorch.zerocheck import _urm  # noqa: E402

_MAGIC = b"FLKURM01"


def _artifacts_dir() -> Path:
    env = os.environ.get("FLOCK_ZORCH_ARTIFACTS")
    return Path(env) if env else Path(__file__).resolve().parents[4] / "artifacts"


class _Reader:
    def __init__(self, raw: bytes):
        self.raw, self.off = raw, 0

    def u64(self) -> int:
        v = int.from_bytes(self.raw[self.off:self.off + 8], "little")
        self.off += 8
        return v

    def f128(self, count: int) -> np.ndarray:
        a = np.frombuffer(self.raw, np.uint64, count * 2, self.off).reshape(count, 2)
        self.off += count * 16
        return a

    def bits(self, count: int) -> np.ndarray:
        a = np.frombuffer(self.raw, np.uint8, count, self.off).copy()
        self.off += count
        return a


def run(path: Path | None = None):
    path = path or (_artifacts_dir() / "gf8_urm_golden.bin")
    raw = path.read_bytes()
    assert raw[:8] == _MAGIC, f"bad magic {raw[:8]!r}"
    rd = _Reader(raw)
    rd.off = 8

    # 1. phi8 table cross-check.
    phi_golden = rd.f128(256)
    if not np.array_equal(_urm.PHI_8_TABLE, phi_golden):
        i = int(np.flatnonzero(np.any(_urm.PHI_8_TABLE != phi_golden, axis=1))[0])
        raise AssertionError(
            f"phi8 table mismatch at {i}: got={_urm.PHI_8_TABLE[i].tolist()} "
            f"golden={phi_golden[i].tolist()}"
        )

    # 2. round1_naive per config.
    configs = []
    for _ in range(rd.u64()):
        m, k_skip = rd.u64(), rd.u64()
        n = 1 << m
        a, b, c = rd.bits(n), rd.bits(n), rd.bits(n)
        r = rd.f128(m)
        ab_golden = rd.f128(1 << k_skip)
        c_golden = rd.f128(1 << k_skip)
        p_ab, p_c = _urm.round1_naive(a, b, c, m, k_skip, r)
        if not np.array_equal(p_ab, ab_golden):
            i = int(np.flatnonzero(np.any(p_ab != ab_golden, axis=1))[0])
            raise AssertionError(
                f"round1_ab mismatch (m={m},k={k_skip}) at {i}: "
                f"got={p_ab[i].tolist()} golden={ab_golden[i].tolist()}"
            )
        if not np.array_equal(p_c, c_golden):
            i = int(np.flatnonzero(np.any(p_c != c_golden, axis=1))[0])
            raise AssertionError(
                f"round1_c mismatch (m={m},k={k_skip}) at {i}: "
                f"got={p_c[i].tolist()} golden={c_golden[i].tolist()}"
            )
        configs.append((m, k_skip))
    return configs


def test_gf8_urm_oracle():
    run()


if __name__ == "__main__":
    cfgs = run()
    print(f"φ₈ + round1 URM byte-match vs flock: PASS (phi8 256-entry; round1 {cfgs})")
