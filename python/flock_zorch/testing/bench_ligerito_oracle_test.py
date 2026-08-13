"""GPU benchmark-profile prove, whole-bundle byte gate vs the flock-challenge
fork — the snark.fast acceptance in one test.

Reproduces the harness worker's instance from (seed, log2) alone via the
device chain (`witgen.blocks_from_seed` → `witness_blake3` →
`lincheck_stripe`), proves under `prover.BENCHMARK_PROFILE` (callback BLAKE3
Fiat-Shamir on the harness domain, BLAKE3 non-root-CV Merkle), serializes
through `proof_io.bundle_bytes`, and compares ALL bytes against
`bench_ligerito_golden.bin` — a bundle produced AND verified by the fork's
own crates (`dump_bench_ligerito`, which runs the harness's acceptance
before writing). Equality means their verifier accepts our proof bytes by
identity. Per-field results localize a divergence before the whole-bundle
check; the circuit constants (cfg, statement digest, CSC rows — all
witness-independent) come from the standard blake3 golden.

Run (regen: cargo run --release --example dump_bench_ligerito):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  FRX_PLATFORMS=cuda,cpu PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/testing/bench_ligerito_oracle_test.py
"""

import argparse
import sys

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)

from flock_zorch import proof_io  # noqa: E402
from flock_zorch.testing._bench_profile import BenchProver  # noqa: E402
from flock_zorch.testing._golden import artifacts_dir  # noqa: E402
from flock_zorch.testing._util import report  # noqa: E402
from flock_zorch.testing.blake3_ligerito_oracle_test import load  # noqa: E402


def _eq(a, b) -> bool:
    """Structural equality over the `parse_bundle` value shapes (dicts, lists,
    numpy arrays, bytes, ints, dataclasses)."""
    if isinstance(a, dict):
        return set(a) == set(b) and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def run(
    golden: str = "blake3_ligerito_golden.bin",
    bundle: str = "bench_ligerito_golden.bin",
    seed: int = 42,
):
    g = load(golden)
    m = g["meta"]["m"]
    want = (artifacts_dir() / bundle).read_bytes()

    # The worker's exact timed body: only the 8-byte seed enters the chain.
    got = BenchProver(g).prove_bundle(seed)

    got_f, want_f = proof_io.parse_bundle(got), proof_io.parse_bundle(want)
    results = [(f"bundle field {k}", _eq(got_f[k], want_f[k])) for k in want_f]
    results.append(("whole bundle bytes", got == want))
    return m, seed, results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--golden",
        default="blake3_ligerito_golden.bin",
        help="circuit-constants golden under artifacts/ (the m-variant dumps)",
    )
    ap.add_argument(
        "--bundle",
        default="bench_ligerito_golden.bin",
        help="fork-verified benchmark-profile bundle under artifacts/",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="harness seed (dump_bench_ligerito's)"
    )
    args = ap.parse_args()
    print(f"device {frx.devices()[0]}")
    m, seed, results = run(args.golden, args.bundle, seed=args.seed)
    return report(
        results,
        f"benchmark-profile prove bundle vs flock-challenge fork "
        f"(m={m} seed={seed})",
    )


if __name__ == "__main__":
    sys.exit(main())
