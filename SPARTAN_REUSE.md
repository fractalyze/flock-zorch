# Spartan reuse experiment

This worktree tests zorch's typed `Stage` design against Flock's byte-identical
Spartan variant. It uses the stage-pipeline zorch worktree through
`.bazelrc.user`.

## Result

Flock cannot reuse zorch's three reference `Spartan` stages without changing its
proof and Fiat–Shamir transcript:

| zorch stage | Flock protocol difference |
| --- | --- |
| `OuterStage` | Flock has a labeled preamble, split challenge sampling, a special univariate-skip first round, ragged messages, and an elided terminal C claim. |
| `InnerStage` | Flock batches A/B matrices with alpha (and sometimes beta), uses a different sumcheck convention and round count, and sends `z_partial`. |
| `WitnessOpenStage` | Flock opens two derived claims through batched ring switching and one combined Ligerito opening, rather than one direct witness evaluation. |

`Spartan` itself also assumes a paired verifier, zorch's `R1CS` representation,
and the exact outputs of those three stages. Flock currently implements only the
prover side, so making it a `Stage` would require a fake verifier.

The useful result is the boundary the design enforces:

- Flock's heterogeneous proof phases use explicit Python dataflow.
- The obsolete carry plus `ProveChain` wrappers disappear.
- A genuine homogeneous recurrence still uses zorch's round machinery:
  `InfProductRound` is driven by `fold_rounds`.
- Flock continues to reuse lower-level zorch transcript, sumcheck-domain,
  ring-switch, and Ligerito components where their contracts really match.

The orchestration refactor removes more code than it adds, but exact reuse of the
reference `Spartan` stages is **0/3**. Generalizing those stages with enough hooks
to accept Flock would move Flock's protocol into strategy callbacks and make the
reference implementation less legible; the shared layer should remain below the
protocol-specific stages.

## Validation

- `//python:zerocheck_oracle_test`: passes
- `//python:lincheck_oracle_test`: passes
- CPU suite: 14/15 pass
- `//python:ring_switch_oracle_test`: fails during import in the overridden
  zorch/FRX environment (`frx.experimental.mosaic.gpu.core` is absent), before
  any Flock prover code executes
- The excluded `e2e_ligerito` fixture/venv is not present in a fresh worktree

