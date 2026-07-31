# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Native unit test for `testing/_util.py`'s `await_all` leaf walk (no golden).

`await_all` is the timing boundary the phase bench measures against: work it
fails to block on is billed to whichever LATER boundary does block, so a phase
split can look plausible while attributing time to the wrong phase. The failure
mode is silent, which is what this test exists to make loud — `tree_leaves`
yields an unregistered dataclass as one opaque leaf, and flock's claims and
proofs are unregistered dataclasses (`register_dataclass` covers the pytrees
that cross a jit boundary, which a claim never does).

Pure host: this is pytree traversal, not field math.
"""
from __future__ import annotations

import dataclasses

import frx
import frx.numpy as fnp
from absl.testing import absltest

from flock_zorch.testing._util import await_all, deep_leaves
from flock_zorch.types import ZerocheckClaim


@dataclasses.dataclass(frozen=True)
class _Inner:
    c: object


@dataclasses.dataclass(frozen=True)
class _Outer:
    a: object
    inner: _Inner
    label: str


def _outer():
    return _Outer(a=fnp.arange(3), inner=_Inner(c=fnp.arange(4)), label="x")


class DeepLeavesTest(absltest.TestCase):
    def test_tree_leaves_alone_misses_dataclass_arrays(self):
        """The premise. If this ever fails because plain dataclasses became
        pytrees, `deep_leaves` is redundant and can go."""
        holder = _outer()
        self.assertEqual(frx.tree_util.tree_leaves(holder), [holder])

    def test_descends_into_nested_dataclasses(self):
        holder = _outer()
        # The `str` field rides along exactly as `tree_leaves` would return it.
        self.assertCountEqual(
            [id(leaf) for leaf in deep_leaves(holder)],
            [id(holder.a), id(holder.inner.c), id(holder.label)],
        )

    def test_finds_a_zerocheck_claim_s_arrays(self):
        """The motivating case: the claim the phase bench hands between phases."""
        vals = [fnp.arange(i + 1) for i in range(6)]
        claim = ZerocheckClaim(
            z=vals[0],
            mlv_challenges=vals[1],
            r_rest=vals[2],
            a_eval=vals[3],
            b_eval=vals[4],
            c_eval=vals[5],
        )
        self.assertCountEqual(
            [id(leaf) for leaf in deep_leaves(claim)], [id(v) for v in vals]
        )

    def test_await_all_returns_its_argument_and_materializes_it(self):
        holder = _outer()
        self.assertIs(await_all(holder), holder)
        self.assertTrue(holder.a.is_ready() and holder.inner.c.is_ready())

    def test_containers_and_none_still_work(self):
        a, b = fnp.arange(3), fnp.arange(4)
        self.assertCountEqual(
            [id(leaf) for leaf in deep_leaves({"x": [a, None], "y": (b,)})],
            [id(a), id(b)],
        )


if __name__ == "__main__":
    absltest.main()
