# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Wire tests for the proof-bundle serializer (no golden).

The format itself is pinned against the fork by the full-prove bundle gate
(a whole serialized proof byte-compared to a bundle the fork's own crates
produced and verified); these tests pin what that gate cannot localize —
that `parse_bundle` is `bundle_bytes`' exact inverse field by field, the
header/enum constants, and the hand-checkable framing of a minimal bundle
(so a shared writer/reader bug cannot hide in the round trip).
"""
from __future__ import annotations

import types

import numpy as np
from absl.testing import absltest

from flock_zorch import proof_io


def _lanes(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**64, size=(n, 2), dtype=np.uint64)


def _root(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=32, dtype=np.uint8)


def _level(seed: int, rows: int, width: int, siblings: int) -> dict:
    return {
        "opened_rows": [_lanes(seed + i, width) for i in range(rows)],
        "merkle_proof": np.stack([_root(seed + 100 + i) for i in range(siblings)]),
    }


def _bundle_pieces():
    zc = types.SimpleNamespace(
        round1_ab=_lanes(1, 4),
        round1_c=_lanes(2, 3),
        multilinear_rounds=[(_lanes(3, 1), _lanes(4, 1)), (_lanes(5, 1), _lanes(6, 1))],
        final_a_eval=_lanes(7, 1),
        final_b_eval=_lanes(8, 1),
        final_c_eval=_lanes(9, 1),
    )
    ligerito = {
        "initial_root": _root(10),
        "initial_proof": _level(11, rows=2, width=8, siblings=3),
        "recursive_roots": [_root(12), _root(13)],
        "recursive_proofs": [_level(14, 1, 4, 2), _level(15, 3, 2, 1)],
        "final_proof": {**_level(16, 1, 2, 1), "yr": _lanes(17, 4)},
        "sumcheck_transcript": [tuple(_lanes(18, 2)), tuple(_lanes(19, 2))],
        "grinding_nonces": [5, 0, 2**63],
        "ood_values": [_lanes(20, 1)[0], _lanes(21, 1)[0]],
        "fold_grinding_nonces": [7],
    }
    params = proof_io.PcsParams(
        m=22,
        log_inv_rate=1,
        log_batch_size=6,
        profile=proof_io.PROFILE_FAST,
        merkle_hash=proof_io.PARAMS_MERKLE_BLAKE3,
    )
    return (
        _root(0),
        params,
        zc,
        [(_lanes(22, 1), _lanes(23, 1))],
        _lanes(24, 5),
        [
            _lanes(25, 6),
            _lanes(26, 6),
        ],
        ligerito,
    )


class BundleRoundTripTest(absltest.TestCase):
    def test_parse_is_the_exact_inverse(self):
        root, params, zc, lc_rounds, z_partial, ring_switches, ligerito = (
            _bundle_pieces()
        )
        data = proof_io.bundle_bytes(
            root, params, zc, lc_rounds, z_partial, ring_switches, ligerito
        )
        out = proof_io.parse_bundle(data)

        np.testing.assert_array_equal(np.frombuffer(out["root"], np.uint8), root)
        self.assertEqual(out["params"], params)
        np.testing.assert_array_equal(out["zc_round1_ab"], zc.round1_ab)
        np.testing.assert_array_equal(out["zc_round1_c"], zc.round1_c)
        np.testing.assert_array_equal(
            out["zc_multilinear"],
            np.stack([np.concatenate(r) for r in zc.multilinear_rounds]),
        )
        np.testing.assert_array_equal(
            out["zc_finals"],
            np.concatenate([zc.final_a_eval, zc.final_b_eval, zc.final_c_eval]),
        )
        np.testing.assert_array_equal(
            out["lc_rounds"], np.stack([np.concatenate(r) for r in lc_rounds])
        )
        np.testing.assert_array_equal(out["lc_z_partial"], z_partial)
        for got, want in zip(out["ring_switches"], ring_switches, strict=True):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(
            np.frombuffer(out["lig_initial_root"], np.uint8),
            ligerito["initial_root"],
        )
        for key, want_levels in (
            ("lig_initial_proof", [ligerito["initial_proof"]]),
            ("lig_recursive_proofs", ligerito["recursive_proofs"]),
            ("lig_final_proof", [ligerito["final_proof"]]),
        ):
            got_levels = out[key] if isinstance(out[key], list) else [out[key]]
            for got, want in zip(got_levels, want_levels, strict=True):
                for gr, wr in zip(got["opened_rows"], want["opened_rows"], strict=True):
                    np.testing.assert_array_equal(gr, wr)
                np.testing.assert_array_equal(got["merkle_proof"], want["merkle_proof"])
        np.testing.assert_array_equal(
            out["lig_final_yr"], ligerito["final_proof"]["yr"]
        )
        np.testing.assert_array_equal(
            out["lig_sumcheck"],
            np.stack([np.stack(r) for r in ligerito["sumcheck_transcript"]]),
        )
        self.assertEqual(out["lig_grinding_nonces"], ligerito["grinding_nonces"])
        np.testing.assert_array_equal(
            out["lig_ood_values"], np.stack(ligerito["ood_values"])
        )
        self.assertEqual(
            out["lig_fold_grinding_nonces"], ligerito["fold_grinding_nonces"]
        )

    def test_header_and_commitment_framing_by_hand(self):
        # Independent of parse_bundle: the first 71 bytes, laid out by hand.
        root, params, zc, lc_rounds, z_partial, ring_switches, ligerito = (
            _bundle_pieces()
        )
        data = proof_io.bundle_bytes(
            root, params, zc, lc_rounds, z_partial, ring_switches, ligerito
        )
        want = (
            b"FLOCK"
            + bytes([4, 2])
            + root.tobytes()
            + (22).to_bytes(8, "little")
            + (1).to_bytes(8, "little")
            + (6).to_bytes(8, "little")
            + (0).to_bytes(4, "little")  # profile Fast
            + (1).to_bytes(4, "little")  # PcsParams merkle_hash: Blake3 = 1
        )
        self.assertEqual(data[: len(want)], want)
        # And the first zerocheck field right after: u64 count then lanes.
        self.assertEqual(
            data[len(want) : len(want) + 8 + 64],
            (4).to_bytes(8, "little") + zc.round1_ab.tobytes(),
        )

    def test_trailing_bytes_are_rejected(self):
        root, params, zc, lc_rounds, z_partial, ring_switches, ligerito = (
            _bundle_pieces()
        )
        data = proof_io.bundle_bytes(
            root, params, zc, lc_rounds, z_partial, ring_switches, ligerito
        )
        with self.assertRaisesRegex(ValueError, "trailing"):
            proof_io.parse_bundle(data + b"\x00")


if __name__ == "__main__":
    absltest.main()
