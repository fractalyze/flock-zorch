"""Byte-match gate for flock's Fiat-Shamir challenger (SHA-256), authored over
zorch's `ByteHashTranscript` (host `HashlibSha256` backend).

Replays the EXACT scripted observe/sample/grind sequence from
`examples/dump_challenger.rs` through `flock_zorch.challenger.Challenger` and
byte-compares every sampled F128 and the grind nonce against the flock golden.
A divergence in any absorbed byte (tag, length prefix, F128 LE order, re-absorb,
PoW) changes a downstream sample, so the samples transitively pin the whole
transcript byte stream.

Run (host-only; needs zorch on PYTHONPATH):
    PYTHONPATH=python:../zorch <venv>/bin/python \
        python/flock_zorch/testing/challenger_oracle_test.py
"""
import os
from pathlib import Path

import numpy as np

from flock_zorch.challenger import Challenger

_MAGIC = b"FLKCHL01"

# Must match examples/dump_challenger.rs exactly.
DOMAIN = b"flock-zorch-oracle"
LABEL = b"flock-zerocheck-v0"
GRIND_BITS = 8


def _artifacts_dir() -> Path:
    env = os.environ.get("FLOCK_ZORCH_ARTIFACTS")
    return Path(env) if env else Path(__file__).resolve().parents[3] / "artifacts"


def _f128(lo: int, hi: int) -> np.ndarray:
    return np.array([lo, hi], dtype=np.uint64)


def _load(path: Path):
    raw = path.read_bytes()
    assert raw[:8] == _MAGIC, f"bad magic {raw[:8]!r} in {path}"
    n = int.from_bytes(raw[8:16], "little")
    samples = np.frombuffer(raw, np.uint64, count=n * 2, offset=16).reshape(n, 2)
    nonce = int.from_bytes(raw[16 + n * 16:16 + n * 16 + 8], "little")
    return samples, nonce


def _replay(make_challenger=Challenger) -> tuple[list[np.ndarray], int]:
    """The scripted sequence — identical to dump_challenger.rs. `make_challenger`
    builds the `Challenger` (its default backend is the host transcript; the device
    gate passes a device-backed factory)."""
    ch = make_challenger(DOMAIN)
    samples: list[np.ndarray] = []

    ch.observe_label(LABEL)
    ch.observe_bytes(bytes(range(32)))
    ch.observe_f128(_f128(0x0123456789ABCDEF, 0xFEDCBA9876543210))
    ch.observe_f128_slice([_f128(1, 0), _f128(2, 0), _f128(0xDEADBEEF, 0xCAFEBABE)])

    s0 = ch.sample_f128()
    samples.append(s0)
    ch.observe_f128(s0)

    sv = ch.sample_f128_vec(5)
    samples.extend(sv)
    ch.observe_f128_slice(sv)

    nonce = ch.grind_pow(GRIND_BITS)

    samples.append(ch.sample_f128())
    samples.append(ch.sample_f128())
    return samples, nonce


def run(path: Path | None = None, *, make_challenger=Challenger):
    path = path or (_artifacts_dir() / "challenger_golden.bin")
    golden, golden_nonce = _load(path)
    samples, nonce = _replay(make_challenger)

    got = np.stack(samples)
    if not np.array_equal(got, golden):
        i = int(np.flatnonzero(np.any(got != golden, axis=1))[0])
        raise AssertionError(
            f"challenger sample mismatch at {i}: got={got[i].tolist()} "
            f"golden={golden[i].tolist()}"
        )
    if nonce != golden_nonce:
        raise AssertionError(f"grind nonce mismatch: got={nonce} golden={golden_nonce}")
    return len(samples), nonce


def test_challenger_oracle():
    run()


if __name__ == "__main__":
    n, nonce = run()
    print(f"challenger byte-match vs flock FsChallenger: PASS ({n} samples, grind nonce={nonce})")
