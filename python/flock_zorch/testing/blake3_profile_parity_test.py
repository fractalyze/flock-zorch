# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`BENCHMARK_PROFILE`'s device transcript against the callback arm it replaced.

The swap is only allowed to change how the transcript is computed, never what
the proof is: the fork's verifier accepts these bytes. So the gate is a full
prove under each arm with every field compared — the same comparison the
harness's own `verified=true` would make, run here where it needs no GPU.

Fiat-Shamir does the work: every challenge binds the whole prefix, so a single
wrong byte in the transcript diverges the proof from that point on and cannot
cancel. Comparing the final fields is therefore comparing the entire transcript
history.

The second test is the one the change exists for. A host-backed transcript
cannot be a jitted loop's carry, so with the callback arm the sumcheck round
loop de-compiles into a host loop — the measured cost is ~10x of the prove at
m32. Whether the loop is *in* the program is a tracing property, so it is
visible here on CPU, and it is the leading indicator: check it before any
wall-clock claim.
"""

import dataclasses
import unittest

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

from flock_zorch import ghash, lincheck, prover  # noqa: E402
from flock_zorch.blake3_challenger import (  # noqa: E402
    Blake3CallbackChallenger,
    Blake3DeviceChallenger,
)
from flock_zorch.hash import merkle  # noqa: E402
from flock_zorch.testing.blake3_ligerito_oracle_test import load  # noqa: E402

GOLDEN = "blake3_ligerito_golden.bin"  # m22, the size the CPU gates run


def _csc(g):
    meta = g["meta"]
    return lincheck.CscCircuit(
        g["a0_rows"], g["b0_rows"], 1 << meta["k_log"], const_pin=meta["const_pin"]
    )


def _prove(challenger_cls):
    """The hash-R1CS path: the golden's a0/b0 are sparse CSC rows, so they ride
    a `CscCircuit` and the dense a0/b0 arguments stay unused."""
    g = load(GOLDEN)
    meta = g["meta"]
    return prover.prove_fast(
        g["z"],
        meta["m"],
        meta["k_log"],
        meta["k_skip"],
        None,
        None,
        g["zlc"],
        g["stmt"],
        g["cfg"],
        circuit=_csc(g),
        domain=b"flock-bench-v0",
        profile=prover.ProveProfile(challenger_cls, merkle.GHASH_BLAKE3_TREE),
    )


def _leaves(obj, path="proof"):
    """Every array in a proof, flattened depth-first with its field path.

    The wire records (`ProveFastResult`, `ZerocheckProof`, …) are plain
    dataclasses rather than pytrees, so `tree_leaves` returns each whole record
    as one opaque leaf and any comparison of it silently degenerates. Walking
    the fields is what makes a mismatch point at a field name.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = []
        for f in dataclasses.fields(obj):
            out += _leaves(getattr(obj, f.name), f"{path}.{f.name}")
        return out
    if isinstance(obj, dict):
        out = []
        for k in sorted(obj):
            out += _leaves(obj[k], f"{path}.{k}")
        return out
    if isinstance(obj, (tuple, list)):
        out = []
        for i, item in enumerate(obj):
            out += _leaves(item, f"{path}[{i}]")
        return out
    if obj is None:
        return []
    if isinstance(obj, (int, float, bool, bytes)):
        return [(path, np.asarray(obj))]
    a = np.asarray(obj)
    if a.dtype == object:
        # An unrecognised container becomes an object array whose comparison is
        # vacuous, so the gate would pass without comparing anything. Fail loudly
        # instead — this flattener has silently degenerated three times.
        raise TypeError(f"{path}: unhandled container {type(obj).__name__}")
    return [(path, ghash._ghash_to_lanes(a) if a.dtype.kind == "V" else a)]


class Blake3ProfileParityTest(unittest.TestCase):
    def test_device_arm_reproduces_the_callback_arm(self):
        want = _leaves(_prove(Blake3CallbackChallenger))
        got = _leaves(_prove(Blake3DeviceChallenger))
        self.assertEqual(
            [p for p, _ in want],
            [p for p, _ in got],
            "proof shape changed, not just bytes",
        )
        self.assertGreater(len(want), 0, "nothing was compared")
        for (path, w), (_, gt) in zip(want, got):
            with self.subTest(field=path):
                np.testing.assert_array_equal(w, gt)

    def test_benchmark_profile_is_the_device_arm(self):
        """The swap itself — without this the parity test above would pass while
        the profile still pointed at the callback arm."""
        self.assertIs(prover.BENCHMARK_PROFILE.challenger_cls, Blake3DeviceChallenger)
        self.assertIs(prover.BENCHMARK_PROFILE.tree, merkle.GHASH_BLAKE3_TREE)

    def test_round_loop_stays_in_the_program(self):
        """The leading indicator: the sumcheck loop must compile INTO the prove
        program under the device arm. Under the callback arm the same trace
        cannot hold it, so `while` count is the thing that separates them —
        not a wall-clock number."""
        from flock_zorch import zerocheck

        g = load(GOLDEN)
        meta = g["meta"]

        def whiles(challenger_cls):
            # `ZerocheckProof` is a wire record, not a pytree, so the traced
            # function returns the arrays rather than the record.
            def run(a, b, c):
                proof, _ = zerocheck.prove_packed(a, b, c, meta["m"], ch=ch)
                return proof.round1_ab, proof.round1_c

            ch = challenger_cls(b"flock-bench-v0")
            return frx.jit(run).lower(g["a"], g["b"], g["z"]).as_text().count("while(")

        dev = whiles(Blake3DeviceChallenger)
        cb = whiles(Blake3CallbackChallenger)
        self.assertGreater(
            dev,
            cb,
            f"device arm kept {dev} while-loops vs callback {cb} — the whole "
            "point of the change is that it keeps more",
        )


if __name__ == "__main__":
    unittest.main()
