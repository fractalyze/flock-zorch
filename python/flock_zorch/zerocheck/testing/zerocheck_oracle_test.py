"""Byte-match gate for flock's zerocheck `prove_packed` — the headline of iter 11
(the first FULL PIOP sub-protocol with a serializable proof).

Replays `prove_packed` on flock's exact witness (`examples/dump_zerocheck.rs`)
and byte-compares every ZerocheckProof field: round1_ab, round1_c, all
multilinear-round (G(1), G(∞)) pairs, and final_{a,b,c}_eval. Cross-checks the
claim's r_rest / z / mlv_challenges first to localize any divergence (the inner-7
constants, the eq-tail, or the host SHA-256 round loop).

Run (jax_enable_x64; zorch on PYTHONPATH):
    PYTHONPATH=python:../zorch <venv>/bin/python \
        python/flock_zorch/zerocheck/testing/zerocheck_oracle_test.py
"""
import os
from pathlib import Path

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch import ghash, zerocheck  # noqa: E402

_MAGIC = b"FLKZC001"
K_SKIP = 6
DOMAIN = b"flock-zc-oracle"


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

    def raw_bytes(self, n: int) -> np.ndarray:
        a = np.frombuffer(self.raw, np.uint8, n, self.off)
        self.off += n
        return a


def _eq(name, got, golden):
    got = ghash.to_lanes(got).reshape(-1, 2)
    golden = np.asarray(golden).reshape(-1, 2)
    if not np.array_equal(got, golden):
        i = int(np.flatnonzero(np.any(got != golden, axis=1))[0])
        raise AssertionError(
            f"{name} mismatch ({len(np.flatnonzero(np.any(got != golden, axis=1)))}/"
            f"{golden.shape[0]}); first @ {i}: got={got[i].tolist()} golden={golden[i].tolist()}"
        )


def run(path: Path | None = None):
    path = path or (_artifacts_dir() / "zerocheck_golden.bin")
    raw = path.read_bytes()
    assert raw[:8] == _MAGIC, f"bad magic {raw[:8]!r}"
    rd = _Reader(raw)
    rd.off = 8
    ell = 1 << K_SKIP

    configs = []
    for _ in range(rd.u64()):
        m = rd.u64()
        n_mlv = m - K_SKIP
        nbytes = (1 << m) // 8
        a = np.unpackbits(rd.raw_bytes(nbytes), bitorder="little")
        b = np.unpackbits(rd.raw_bytes(nbytes), bitorder="little")
        c = np.unpackbits(rd.raw_bytes(nbytes), bitorder="little")
        round1_ab = rd.f128(ell)
        round1_c = rd.f128(ell)
        assert rd.u64() == n_mlv
        rounds = rd.f128(2 * n_mlv)  # interleaved m1, mi
        final_a, final_b, final_c = rd.f128(1)[0], rd.f128(1)[0], rd.f128(1)[0]
        z = rd.f128(1)[0]
        mlv_ch = rd.f128(n_mlv)
        r_rest = rd.f128(m - K_SKIP)
        _a_eval, _b_eval, _c_eval = rd.f128(1)[0], rd.f128(1)[0], rd.f128(1)[0]

        out, _transcript = zerocheck.prove_packed(a, b, c, m, DOMAIN)

        # Localization cross-checks (claim) first.
        _eq(f"r_rest(m={m})", out.r_rest, r_rest)
        _eq(f"z(m={m})", ghash.from_ghash_host(out.z), z)
        _eq(f"mlv_challenges(m={m})", out.mlv_challenges, mlv_ch)

        # The proof — the actual gate.
        _eq(f"round1_ab(m={m})", out.round1_ab, round1_ab)
        _eq(f"round1_c(m={m})", out.round1_c, round1_c)
        got_rounds = np.stack([v for pair in out.multilinear_rounds for v in pair])
        _eq(f"multilinear_rounds(m={m})", got_rounds, rounds)
        _eq(f"final_a_eval(m={m})", out.final_a_eval, final_a)
        _eq(f"final_b_eval(m={m})", out.final_b_eval, final_b)
        _eq(f"final_c_eval(m={m})", out.final_c_eval, final_c)
        configs.append(m)
    return configs


def test_zerocheck_oracle():
    run()


if __name__ == "__main__":
    cfgs = run()
    print(f"zerocheck prove_packed byte-match vs flock (software mul): PASS (m={cfgs})")
