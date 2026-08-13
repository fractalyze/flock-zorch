# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`clmad_ptxas_verdict` against real `ptxas --version` output.

Pure-text cases — no GPU, no subprocess — because the verdict is the one part
of the guard whose wrong answer is silent: a false pass measures the shift/XOR
fallback as a regression, a false refusal blocks every card that has no clmad
to begin with.
"""

from __future__ import annotations

from absl.testing import absltest

from flock_zorch.testing._ptxas import clmad_ptxas_verdict

_V12_9 = (
    "Cuda compilation tools, release 12.9, V12.9.86\n"
    "Build cuda_12.9.r12.9/compiler.36037853_0\n"
)
_V13_3 = (
    "Cuda compilation tools, release 13.3, V13.3.33\n"
    "Build cuda_13.3.r13.3/compiler.37862127_0\n"
)


class ClmadPtxasVerdictTest(absltest.TestCase):
    def test_pre_clmad_ptxas_on_a_clmad_card_refuses(self) -> None:
        reason = clmad_ptxas_verdict(_V12_9, "12.0")
        self.assertIsNotNone(reason)
        # The message must name both sides of the mismatch, or the reader
        # cannot act on it.
        self.assertIn("12.9", reason)
        self.assertIn("13.3", reason)

    def test_clmad_ptxas_passes(self) -> None:
        self.assertIsNone(clmad_ptxas_verdict(_V13_3, "12.0"))

    def test_pre_clmad_card_never_blocks(self) -> None:
        # Hopper/Ampere have no clmad at any toolchain: the fallback IS the
        # measurement there, so a refusal would block every run on those cards.
        self.assertIsNone(clmad_ptxas_verdict(_V12_9, "9.0"))
        self.assertIsNone(clmad_ptxas_verdict(_V12_9, "8.0"))

    def test_unknowns_never_block(self) -> None:
        self.assertIsNone(clmad_ptxas_verdict(None, "12.0"))
        self.assertIsNone(clmad_ptxas_verdict(_V12_9, None))
        self.assertIsNone(clmad_ptxas_verdict("not a ptxas banner", "12.0"))
        self.assertIsNone(clmad_ptxas_verdict(_V12_9, "sm_120a"))


if __name__ == "__main__":
    absltest.main()
