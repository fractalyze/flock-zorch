"""End-to-end fused-prover byte gate on the DEVICE SHA-256 transcript.

Re-runs `e2e_oracle_test.run` (the full flock `prover::prove` byte-compare, stages
A–F against `artifacts/e2e_golden.bin`) with the Fiat-Shamir challenger backed by
zorch's `DeviceSha256Transcript` (SHA-256 on the `zorch.sha256` marker, no host
`hashlib`) instead of the host `Sha256Transcript`. Because both transcripts share
the identical Merlin byte framing, every stage — commit, zerocheck, lincheck, the
ab/c claims, and the batched PCS open — must byte-match the same golden. This pins
the device transcript through the WHOLE prover, not just the challenger-replay KAT
(`challenger_device_oracle_test`).

MANUAL gate — not in the CI oracle-gates loop. The device byte transcript
re-hashes its whole growing buffer on device per squeeze, so a full prove runs
orders of magnitude slower than the host transcript (minutes, not seconds); that
is too heavy for every-PR CI. `challenger_device_oracle_test` (fast, in CI) guards
the transcript's byte framing on every PR; run this heavier full-prover check by
hand on GPU when the transcript or prover wiring changes.

Run (needs the device transcript on PYTHONPATH — the pinned third_party/zorch
submodule, or a dev zorch checkout, that includes `device_byte_transcript`):
    PYTHONPATH=python:third_party/zorch <venv>/bin/python \
        python/flock_zorch/testing/e2e_device_oracle_test.py
"""
import sys
import unittest

from flock_zorch.testing.e2e_oracle_test import MUL, run

try:
    from zorch.device_byte_transcript import DeviceSha256Transcript

    _HAVE_DEVICE = True
except ImportError:  # pinned zorch predates the device transcript
    _HAVE_DEVICE = False


def run_device():
    return run(MUL, transcript_cls=DeviceSha256Transcript)


def test_e2e_oracle_device():
    if not _HAVE_DEVICE:
        raise unittest.SkipTest(
            "zorch.device_byte_transcript not importable (pin predates flock-zorch#6)"
        )
    _m, results = run_device()
    for name, ok in results:
        assert ok, f"device-transcript e2e stage mismatch: {name}"


def main() -> int:
    if not _HAVE_DEVICE:
        raise SystemExit(
            "zorch.device_byte_transcript not on PYTHONPATH; point it at a zorch "
            "checkout that includes flock-zorch#6"
        )
    m, results = run_device()
    allok = True
    for nm, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}")
        allok = allok and ok
    print(f"DEVICE-transcript e2e byte-match vs flock prove (identity m={m}): "
          f"{'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
