"""Byte-match gate for the DEVICE Fiat-Shamir challenger.

Replays the exact `examples/dump_challenger.rs` script through
`flock_zorch.challenger.Challenger` backed by zorch's device `Sha256` byte hash
(SHA-256 on the `zorch.sha256` marker, no host hashlib) and byte-compares every
sampled F128 and the grind nonce against the same flock golden the host gate
uses. This is the direct bit-for-bit anchor to flock-core's `FsChallenger` for
the on-device transcript (flock-zorch#6): the samples transitively pin every
absorbed byte (tag, length prefix, F128 LE order, re-absorb, PoW).

Run (needs zorch on PYTHONPATH — the pinned third_party/zorch submodule or a dev
zorch checkout):
    PYTHONPATH=python:third_party/zorch <venv>/bin/python \
        python/flock_zorch/testing/challenger_device_oracle_test.py
"""
from flock_zorch.challenger import Challenger
from flock_zorch.testing.challenger_oracle_test import run
from zorch.hash.sha256 import Sha256


def _device_challenger(domain: bytes) -> Challenger:
    return Challenger(domain, byte_hash=Sha256())


def run_device():
    return run(make_challenger=_device_challenger)


def test_challenger_oracle_device():
    run_device()


if __name__ == "__main__":
    n, nonce = run_device()
    print(f"DEVICE challenger byte-match vs flock FsChallenger: PASS "
          f"({n} samples, grind nonce={nonce})")
