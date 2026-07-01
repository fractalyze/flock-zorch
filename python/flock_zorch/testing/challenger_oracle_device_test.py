"""Byte-match gate for the DEVICE Fiat-Shamir challenger.

Replays the exact `examples/dump_challenger.rs` script through
`flock_zorch.challenger.Challenger` backed by zorch's *device*
`DeviceSha256Transcript` (SHA-256 on the `zorch.sha256` marker, no host hashlib)
and byte-compares every sampled F128 and the grind nonce against the same flock
golden the host gate uses. This is the direct bit-for-bit anchor to flock-core's
`FsChallenger` for the on-device transcript (flock-zorch#6): the samples
transitively pin every absorbed byte (tag, length prefix, F128 LE order,
re-absorb, PoW).

Run (needs the device transcript on PYTHONPATH — the dev zorch checkout, or a
pin that includes `device_byte_transcript`):
    PYTHONPATH=python:../zorch <venv>/bin/python \
        python/flock_zorch/testing/challenger_oracle_device_test.py
"""
import unittest

from flock_zorch.challenger import Challenger
from flock_zorch.testing.challenger_oracle_test import run

try:
    from zorch.device_byte_transcript import DeviceSha256Transcript

    _HAVE_DEVICE = True
except ImportError:  # pinned zorch predates the device transcript
    _HAVE_DEVICE = False


def _device_challenger(domain: bytes) -> Challenger:
    return Challenger(domain, transcript_cls=DeviceSha256Transcript)


def run_device():
    return run(make_challenger=_device_challenger)


def test_challenger_oracle_device():
    if not _HAVE_DEVICE:
        raise unittest.SkipTest(
            "zorch.device_byte_transcript not importable (pin predates flock-zorch#6)"
        )
    run_device()


if __name__ == "__main__":
    if not _HAVE_DEVICE:
        raise SystemExit(
            "zorch.device_byte_transcript not on PYTHONPATH; point it at a zorch "
            "checkout that includes flock-zorch#6"
        )
    n, nonce = run_device()
    print(f"DEVICE challenger byte-match vs flock FsChallenger: PASS "
          f"({n} samples, grind nonce={nonce})")
