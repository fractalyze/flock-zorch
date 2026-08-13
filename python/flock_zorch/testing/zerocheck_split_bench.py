# Copyright 2026 The Flock-Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Split the zerocheck phase into its sub-steps and time each one.

`prove_phase_bench` stops at `commit / zerocheck / lincheck / open`. Once
zerocheck dominates that row stops being actionable: it hides a round-1 URM and
a multilinear ladder with very different shapes.

This drills one level in without re-deriving anything. It wraps
`zerocheck.prove_packed` for the duration of one real prove — same goldens, same
witness, same Fiat-Shamir state as the byte gates — and inside the wrapper calls
the prover's OWN round objects (`_UrmRound`, `_MultilinearRound`) one at a time.
A harness that re-implements the rounds stops matching the prover it claims to
measure; this one cannot, and it proves so by byte-comparing the proof it
reassembles against an unsplit `prove_packed` on the same transcript. A mismatch
aborts rather than warning.

`--inner` is a different kind of number and is labelled as such: the multilinear
round is one fused device program, so its parts are re-timed standalone on the
same inputs. They answer "which piece is big", not "where did the round's
milliseconds go". Which steps they are depends on the ladder the circuit takes,
so they are named by `_inner_split` rather than fixed here.

Rank targets by *marginal* µs/hash across an m step, never by share at one m —
`docs/measurement.md` is emphatic about this and the ranking inverts between
m=22 and m=28. Pass `--compare` with a smaller-m golden to get that column.

Run:

    PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \\
        python/flock_zorch/testing/zerocheck_split_bench.py blake3 \\
            --golden blake3_ligerito_golden_m28.bin \\
            --compare blake3_ligerito_golden_m26.bin --inner
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import frx

# Must precede every flock_zorch import: module-level ghash constants (e.g.
# sumcheck.eq._ONE_G) are built at import time and silently truncate to uint32
# without it.
frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402

from flock_zorch import zerocheck  # noqa: E402
from flock_zorch.sumcheck.eq import _ONE_G  # noqa: E402
from flock_zorch.testing._util import await_all, best_of  # noqa: E402
from flock_zorch.testing.prove_phase_bench import (  # noqa: E402
    CIRCUITS,
    Circuit,
    make_prove,
)
from flock_zorch.types import ZerocheckProof  # noqa: E402
from flock_zorch.zerocheck._fold import _fold_at_z, _lagrange_weights  # noqa: E402
from flock_zorch.zerocheck.prover import (  # noqa: E402
    _EQ_TABLES,
    _EQ_TABLES_SQ,
    _MEDIUM_G,
    _SMALL_G,
    _SQRT,
    K_SKIP,
    _mlv_sumcheck,
    _mlv_sumcheck_sq,
    _MultilinearRound,
    _UrmRound,
    _ZerocheckCarry,
    equal_factors,
    sample_challenge_coords,
)

SUB_STEPS = ("urm_round1", "mlv_round")


def _timed(fn, runs: int):
    """Warmup-excluded best-of-`runs` ms AND the fastest run's own value.

    `_util.best_of` is warmup-excluded because every step here is a fresh
    jitted program on first call and at m=28 the compile dominates: timing
    without the warm-up read ~2x steady state, and the two m values warmed
    unequally, which corrupts the marginal column.

    It also already keeps (and has awaited) the fastest run's result, so a
    caller never has to re-run a timed program just to bind its value — at
    m=28 each such re-run is another full-size device program.
    """

    def once():
        value = fn()
        return value, value

    return best_of(once, runs)


def _timed_round(step, carry, ch, runs: int):
    """Time `step` best-of-`runs` and carry the fastest run's own outputs.

    Sha256Challenger replaces `_t` rather than mutating it, so snapshotting the
    field is enough to rewind the transcript between repeats — without that,
    each repeat would advance Fiat-Shamir and the reassembled proof would not
    match. Every repeat therefore does identical work from identical state and
    lands identical outputs, which is what lets the fastest run's be the ones
    threaded onward instead of running the whole round once more for real.
    """
    saved = ch._t

    def once():
        ch._t = saved
        out = step(carry, ch)
        return out[0], out

    return best_of(once, runs)


def _proof_from_carry(carry) -> ZerocheckProof:
    return ZerocheckProof(
        round1_ab=carry.round1_ab,
        round1_c=carry.round1_c,
        multilinear_rounds=carry.multilinear_rounds,
        final_a_eval=carry.final_a_eval,
        final_b_eval=carry.final_b_eval,
        final_c_eval=carry.final_c_eval,
    )


def _wire_bytes(proof: ZerocheckProof) -> bytes:
    """Every array the proof carries, flattened in a fixed order.

    Round messages are tuples of arrays, so this recurses rather than assuming
    one array per field.
    """
    out = bytearray()

    def visit(part):
        if isinstance(part, (list, tuple)):
            for item in part:
                visit(item)
        else:
            out.extend(frx.device_get(part).tobytes())

    visit(
        [
            proof.round1_ab,
            proof.round1_c,
            proof.final_a_eval,
            proof.final_b_eval,
            proof.final_c_eval,
            proof.multilinear_rounds,
        ]
    )
    return bytes(out)


def _inner_split(carry, transcript, times, runs: int) -> list[str]:
    """Re-time the multilinear round's parts standalone, on the ladder
    `equal_factors` says the prover takes. Returns the step names it filled.

    The two ladders run different programs and so have different step names;
    asking `prover.equal_factors` rather than re-deriving the choice is what
    keeps this from timing a program the circuit never runs (see that helper).

    See module docstring: these are standalone re-timings and still answer
    "which piece is big" rather than "where did the round's milliseconds go".
    """
    steps: list[str] = []

    def timed(name, fn):
        """Time `fn` under `name`, record the step, and hand back its value."""
        times[name], value = _timed(fn, runs)
        steps.append(name)
        return value

    weights = _lagrange_weights(K_SKIP, carry.z, 0)
    r_tail = carry.r[K_SKIP + 1 :]
    a_g = timed("fold_at_z", lambda: _fold_at_z(carry.a_rows, weights))

    # The ladder threads the transcript; hand it the same pre-round state each
    # repeat so every timing measures identical work.
    if equal_factors(carry):

        def sqrt_then_tables():
            cs = _SQRT(r_tail)
            return cs, _EQ_TABLES_SQ(cs)

        cs_sqrt, eq_tables = timed("sqrt+eq_tables", sqrt_then_tables)
        timed(
            "mlv_ladder",
            lambda: _mlv_sumcheck_sq(a_g, eq_tables, cs_sqrt, _ONE_G, transcript),
        )
    else:
        b_g = timed("fold_at_z_b", lambda: _fold_at_z(carry.b_rows, weights))
        eq_tables = timed("eq_tables", lambda: _EQ_TABLES(r_tail))
        timed(
            "mlv_ladder",
            lambda: _mlv_sumcheck(a_g, b_g, eq_tables, _ONE_G, transcript),
        )
    return steps


def make_split_prove_packed(unsplit, times: dict, runs: int, inner: bool):
    """A `prove_packed` drop-in that times the rounds and self-checks.

    `unsplit` is the original callable, captured before the patch — reaching for
    the module attribute here would recurse into this wrapper.
    """

    def split_prove_packed(a_bits, b_bits, c_bits, m, domain=None, ch=None):
        assert ch is not None, "the bench always threads a shared challenger"
        entry = ch._t

        # Reference: the unsplit path on this exact transcript.
        ref_proof, ref_claim = unsplit(a_bits, b_bits, c_bits, m, domain, ch)
        await_all(ref_proof)
        ref_bytes = _wire_bytes(ref_proof)

        ch._t = entry
        r_skip, r_outer = sample_challenge_coords(ch, m, K_SKIP)
        r = fnp.concatenate([r_skip, _SMALL_G, _MEDIUM_G, r_outer])
        carry = _ZerocheckCarry(a_bits, b_bits, c_bits, r=r)

        ms, (carry, ch, _) = _timed_round(_UrmRound(m, K_SKIP), carry, ch, runs)
        times["urm_round1"] = ms
        pre_mlv_transcript = ch._t
        ms, (carry, ch, _) = _timed_round(_MultilinearRound(m, K_SKIP), carry, ch, runs)
        times["mlv_round"] = ms

        if _wire_bytes(_proof_from_carry(carry)) != ref_bytes:
            raise SystemExit(
                "zerocheck split does not reproduce prove_packed's wire bytes — "
                "the harness has drifted from the prover; fix it before "
                "trusting any number it prints"
            )

        if inner:
            times["inner_steps"] = _inner_split(carry, pre_mlv_transcript, times, runs)

        # Hand back the reference objects so the rest of the prove is unaffected.
        return ref_proof, ref_claim

    return split_prove_packed


def run(circ: Circuit, golden: str | None, args) -> dict:
    g = circ.ingest(golden)
    meta = g["meta"]
    times: dict = {}

    original = zerocheck.prove_packed
    zerocheck.prove_packed = make_split_prove_packed(
        original, times, args.runs, args.inner
    )
    try:
        make_prove(circ, g, args.unpacked, seed=args.seed)()
    finally:
        zerocheck.prove_packed = original

    times["hashes"] = circ.hashes_per_proof(meta)
    times["m"] = meta["m"]
    return times


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("circuit", choices=sorted(CIRCUITS))
    parser.add_argument("--golden", default=None)
    parser.add_argument(
        "--compare",
        default=None,
        help="second (smaller m) golden; adds the marginal us/hash column",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--inner", action="store_true")
    parser.add_argument("--unpacked", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    circ = Circuit(args.circuit)
    big = run(circ, args.golden, args)
    small = run(circ, args.compare, args) if args.compare else None

    steps = list(SUB_STEPS)
    if args.inner:
        if small and small["inner_steps"] != big["inner_steps"]:
            raise SystemExit(
                "the two goldens took different multilinear ladders "
                f"({big['inner_steps']} vs {small['inner_steps']}) — the "
                "marginal column would subtract unlike programs"
            )
        steps += big["inner_steps"]
    header = f"{'step':<16}{'m=' + str(big['m']):>11}"
    if small:
        header += f"{'m=' + str(small['m']):>11}{'marg us/hash':>14}"
    print(f"\n{header}\n{'-' * len(header)}")
    for step in steps:
        row = f"{step:<16}{big[step]:>9.2f}ms"
        if small:
            dh = big["hashes"] - small["hashes"]
            row += f"{small[step]:>9.2f}ms"
            row += f"{(big[step] - small[step]) * 1e3 / dh:>13.2f}"
        print(row)
    if args.inner:
        print("\n(inner steps are standalone re-timings; they do not sum to mlv_round)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
