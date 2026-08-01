"""Per-opcode GF(2^128) census for the flock prover.

Run on an otherwise idle GPU.  The benchmark sweeps powers of two so launch
cost can be distinguished from a bad steady-state lowering, and compares the
native field operations with uint64-lane equivalents where that comparison is
meaningful.  JSON output is intended to be attached to performance issues.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)
import frx.numpy as fnp  # noqa: E402
from zorch.pcs.ring_switch import (  # noqa: E402
    bit_slice_evals,
    tensor_algebra_transpose,
)

from flock_zorch import ghash  # noqa: E402


ArrayFn = Callable[..., Any]
_GHASH = fnp.binary_field_ghash


@dataclass(frozen=True)
class Roofline:
    """Per-element traffic and carryless-multiply work lower bounds."""

    bytes_per_element: float
    clmul_per_element: float = 0.0
    integer_ops_per_element: float = 0.0


@dataclass(frozen=True)
class Point:
    log_elements: int
    elements: int
    milliseconds: float
    ns_per_element: float
    billion_elements_per_second: float
    effective_gbps: float
    roofline_percent: float | None
    lane_milliseconds: float | None
    native_over_lane: float | None


@dataclass(frozen=True)
class Census:
    primitive: str
    classification: str
    roofline: Roofline
    points: list[Point]
    count_per_prove: int | None
    estimated_ms_per_prove: float | None
    estimated_gap_ms_per_prove: float | None


@dataclass(frozen=True)
class Operation:
    name: str
    roofline: Callable[[int], Roofline]
    prepare: Callable[[int], tuple[ArrayFn, tuple[Any, ...], ArrayFn | None, tuple[Any, ...]]]


def _random_lanes(n: int, seed: int) -> Any:
    return fnp.asarray(
        np.random.default_rng(seed).integers(0, 2**64, (n, 2), dtype=np.uint64)
    )


def _ghash_args(n: int) -> tuple[Any, Any, Any, Any]:
    a_l = _random_lanes(n, 1)
    b_l = _random_lanes(n, 2)
    return ghash.to_ghash(a_l), ghash.to_ghash(b_l), a_l, b_l


def _elementwise(
    n: int, native: ArrayFn, lanes: ArrayFn | None
) -> tuple[ArrayFn, tuple[Any, ...], ArrayFn | None, tuple[Any, ...]]:
    a, b, a_l, b_l = _ghash_args(n)
    return native, (a, b), lanes, (a_l, b_l) if lanes else ()


def _add(n: int):
    return _elementwise(n, lambda a, b: a + b, lambda a, b: a ^ b)


def _sub(n: int):
    # Subtraction is XOR in characteristic two.  Keep a separate census row:
    # front-end opcode selection can still make the lowerings diverge.
    return _elementwise(n, lambda a, b: a - b, lambda a, b: a ^ b)


def _multiply(n: int):
    return _elementwise(n, lambda a, b: a * b, None)


def _xor_sum(n: int):
    a, _, a_l, _ = _ghash_args(n)
    return (
        lambda x: fnp.sum(x, axis=0),
        (a,),
        lambda x: frx.lax.reduce_xor(x, (0,)),
        (a_l,),
    )


def _select(n: int):
    a, b, a_l, b_l = _ghash_args(n)
    mask = fnp.asarray(np.random.default_rng(3).integers(0, 2, n, dtype=np.uint8)).astype(
        bool
    )
    return (
        lambda m, x, y: fnp.where(m, x, y),
        (mask, a, b),
        lambda m, x, y: fnp.where(m[:, None], x, y),
        (mask, a_l, b_l),
    )


def _indices(n: int) -> Any:
    # A permutation makes scatter-set byte-comparable and gives both operations
    # the non-coalesced access pattern that matters to the prover.
    return fnp.asarray(np.random.default_rng(4).permutation(n).astype(np.int32))


def _gather(n: int):
    a, _, a_l, _ = _ghash_args(n)
    indices = _indices(n)
    return lambda x, i: x[i], (a, indices), lambda x, i: x[i], (a_l, indices)


def _scatter(n: int):
    a, _, a_l, _ = _ghash_args(n)
    indices = _indices(n)
    return (
        lambda x, i: ghash.zeros(x.shape[0]).at[i].set(x),
        (a, indices),
        lambda x, i: fnp.zeros_like(x).at[i].set(x),
        (a_l, indices),
    )


def _inverse(n: int):
    a, _, _, _ = _ghash_args(n)
    one = ghash.to_ghash(fnp.asarray(np.array([1, 0], np.uint64)))
    # This is the primitive emitted by the current FRX prover.  flock-core uses
    # a chunked Montgomery pass instead; the census deliberately exposes that
    # algorithmic gap instead of timing a large source-level reimplementation.
    # Adding one avoids the sole non-invertible field element without a cast.
    a = a + one
    return lambda x, o: o / x, (a, one), None, ()


def _additive_ntt(n: int):
    a, _, _, _ = _ghash_args(n)
    return (
        lambda x: frx.lax.ntt(x, ntt_type="NTT", ntt_length=x.shape[0]),
        (a,),
        None,
        (),
    )


def _additive_ntt_extend(n: int):
    a, _, _, _ = _ghash_args(n)
    rows = a.reshape(-1, 64)

    def extend(x):
        # flock zerocheck's round-1 S -> Lambda extension: a growing batch of
        # fixed-size INTT_64 -> cosetNTT_64 transforms, not one length-n NTT.
        coeffs = frx.lax.ntt(x, ntt_type="INTT", ntt_length=64)
        return frx.lax.ntt(
            coeffs, ntt_type="NTT", ntt_length=64, coset=64
        )

    return extend, (rows,), None, ()


def _ring_switch(n: int):
    selectors, values, _, _ = _ghash_args(n)
    return bit_slice_evals, (selectors, values), None, ()


def _ring_switch_transpose(n: int):
    values, _, _, _ = _ghash_args(n)
    rows = values.reshape(-1, 128)
    return frx.vmap(tensor_algebra_transpose), (rows,), None, ()


def _to_ghash(n: int):
    lanes = _random_lanes(n, 1)
    return ghash.to_ghash, (lanes,), None, ()


def _from_ghash(n: int):
    lanes = _random_lanes(n, 1)
    return ghash.from_ghash, (ghash.to_ghash(lanes),), None, ()


def _constant_roofline(
    byte_count: float, clmul_count: float = 0.0, integer_count: float = 0.0
):
    return lambda _n: Roofline(byte_count, clmul_count, integer_count)


def _ntt_roofline(n: int, transforms: int = 1) -> Roofline:
    # XLA's additive emitter fuses at most nine butterfly layers per pass.  A
    # pass reads and writes each 16-byte element once; arithmetic still scales
    # with every layer.  This is the actual emitter traffic lower bound, not
    # the much looser one-global-round-trip-per-butterfly model.
    layers = int(math.log2(n))
    passes = math.ceil(layers / 9)
    # Each butterfly performs one field multiply for two elements, and the
    # CUDA lowering implements one GF(2^128) multiply with eight clmad ops.
    clmads = 4.0 * layers * transforms
    return Roofline(32.0 * passes * transforms, clmads)


OPERATIONS = (
    Operation("add", _constant_roofline(48), _add),
    Operation("sub", _constant_roofline(48), _sub),
    Operation("multiply", _constant_roofline(48, 8), _multiply),
    Operation("additive_ntt", _ntt_roofline, _additive_ntt),
    Operation(
        "additive_ntt_extend",
        # Both six-layer transforms fit in one fused emitter pass.
        lambda _n: Roofline(64, 48),
        _additive_ntt_extend,
    ),
    Operation("xor_sum_reduce", _constant_roofline(16), _xor_sum),
    Operation("select", _constant_roofline(49), _select),
    Operation("gather", _constant_roofline(36), _gather),
    Operation("scatter", _constant_roofline(36), _scatter),
    # Fermat inverse is 127 square + 126 multiply, at eight clmads apiece.
    Operation("batch_inverse", _constant_roofline(32, 253 * 8), _inverse),
    Operation(
        "ring_switch_bit_slices", _constant_roofline(32, integer_count=512), _ring_switch
    ),
    Operation(
        "ring_switch_transpose",
        _constant_roofline(32, integer_count=128),
        _ring_switch_transpose,
    ),
    # Same bytes and layout: the operation itself has zero traffic.  A
    # standalone jit result may still expose an entry-boundary copy.
    Operation("to_ghash_bitcast", _constant_roofline(0), _to_ghash),
    Operation("from_ghash_bitcast", _constant_roofline(0), _from_ghash),
)


def _time(fn: ArrayFn, args: tuple[Any, ...], warmups: int, iterations: int) -> float:
    compiled = frx.jit(fn)
    result = compiled(*args)
    frx.block_until_ready(result)
    for _ in range(warmups):
        result = compiled(*args)
    frx.block_until_ready(result)
    start = time.perf_counter()
    for _ in range(iterations):
        result = compiled(*args)
    frx.block_until_ready(result)
    return (time.perf_counter() - start) / iterations


def roofline_ns_per_element(
    roofline: Roofline,
    bandwidth_gbps: float | None,
    clmul_gops: float | None,
    integer_gops: float | None = None,
) -> float | None:
    # A clmul-bearing opcode has no complete roofline until its issue ceiling
    # is supplied.  Reporting only the memory bound would falsely diagnose a
    # compute-bound multiply as a lowering defect.
    if roofline.clmul_per_element and not clmul_gops:
        return None
    if roofline.integer_ops_per_element and not integer_gops:
        return None
    bounds = []
    if bandwidth_gbps:
        bounds.append(roofline.bytes_per_element / bandwidth_gbps)
    if clmul_gops and roofline.clmul_per_element:
        bounds.append(roofline.clmul_per_element / clmul_gops)
    if integer_gops and roofline.integer_ops_per_element:
        bounds.append(roofline.integer_ops_per_element / integer_gops)
    return max(bounds) if bounds else None


def classify(points: list[Point], collapse_ratio: float, loser_ratio: float) -> str:
    """Classify only measured evidence; fast flat kernels are not called defects."""
    last = points[-1]
    dispatch = (
        len(points) > 1
        and points[0].ns_per_element / last.ns_per_element >= collapse_ratio
    )
    if last.native_over_lane is not None:
        defect = last.native_over_lane >= loser_ratio
    else:
        defect = (
            last.roofline_percent is not None
            and last.roofline_percent <= 100 / loser_ratio
        )
    if defect and dispatch:
        return "lowering-defect+dispatch-bound"
    if defect:
        return "lowering-defect"
    if dispatch:
        return "dispatch-bound"
    return "efficient-or-inconclusive"


def _reference_ns(point: Point, roofline_ns: float | None) -> float:
    references = [v for v in (roofline_ns,) if v is not None]
    if point.native_over_lane and point.lane_milliseconds is not None:
        references.append(point.lane_milliseconds * 1e6 / point.elements)
    return max(references, default=point.ns_per_element)


def run_operation(
    operation: Operation,
    logs: range,
    *,
    warmups: int,
    iterations: int,
    bandwidth_gbps: float | None,
    clmul_gops: float | None,
    integer_gops: float | None,
    collapse_ratio: float,
    loser_ratio: float,
    count_per_prove: int | None,
) -> Census:
    points = []
    for log_n in logs:
        n = 1 << log_n
        native, args, lane, lane_args = operation.prepare(n)
        seconds = _time(native, args, warmups, iterations)
        lane_seconds = _time(lane, lane_args, warmups, iterations) if lane else None
        roofline = operation.roofline(n)
        ideal_ns = roofline_ns_per_element(
            roofline, bandwidth_gbps, clmul_gops, integer_gops
        )
        ns = seconds * 1e9 / n
        points.append(
            Point(
                log_elements=log_n,
                elements=n,
                milliseconds=seconds * 1e3,
                ns_per_element=ns,
                billion_elements_per_second=1 / ns,
                effective_gbps=roofline.bytes_per_element / ns,
                roofline_percent=(100 * ideal_ns / ns if ideal_ns else None),
                lane_milliseconds=(lane_seconds * 1e3 if lane_seconds else None),
                native_over_lane=(seconds / lane_seconds if lane_seconds else None),
            )
        )
        del args, lane_args
        gc.collect()

    final = points[-1]
    final_roofline = operation.roofline(final.elements)
    ideal_ns = roofline_ns_per_element(
        final_roofline, bandwidth_gbps, clmul_gops, integer_gops
    )
    estimate = (
        final.ns_per_element * count_per_prove / 1e6
        if count_per_prove is not None
        else None
    )
    gap = (
        max(0.0, final.ns_per_element - _reference_ns(final, ideal_ns))
        * count_per_prove
        / 1e6
        if count_per_prove is not None
        else None
    )
    return Census(
        primitive=operation.name,
        classification=(
            "entry-boundary-copy"
            if final_roofline.bytes_per_element == 0
            else classify(points, collapse_ratio, loser_ratio)
        ),
        roofline=final_roofline,
        points=points,
        count_per_prove=count_per_prove,
        estimated_ms_per_prove=estimate,
        estimated_gap_ms_per_prove=gap,
    )


def _load_counts(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not all(
        isinstance(k, str) and isinstance(v, int) and v >= 0 for k, v in raw.items()
    ):
        raise ValueError("--op-counts must be a JSON object of non-negative integers")
    unknown = raw.keys() - {operation.name for operation in OPERATIONS}
    if unknown:
        raise ValueError(f"--op-counts contains unknown primitives: {sorted(unknown)}")
    return raw


def _print_table(results: list[Census]) -> None:
    print("\nprimitive                    class                       ns/elem   roof%   lane×")
    for result in results:
        point = result.points[-1]
        roof = "--" if point.roofline_percent is None else f"{point.roofline_percent:6.1f}"
        lane = "--" if point.native_over_lane is None else f"{point.native_over_lane:6.2f}"
        print(
            f"{result.primitive:28} {result.classification:26} "
            f"{point.ns_per_element:8.3f} {roof:>7} {lane:>7}"
        )
    ranked = [r for r in results if r.estimated_gap_ms_per_prove is not None]
    if ranked:
        print("\nranked contribution (steady-state estimate)")
        for rank, result in enumerate(
            sorted(ranked, key=lambda r: r.estimated_gap_ms_per_prove or 0, reverse=True), 1
        ):
            print(
                f"{rank:2}. {result.primitive:28} "
                f"{result.estimated_ms_per_prove:9.3f} ms total  "
                f"{result.estimated_gap_ms_per_prove:9.3f} ms gap"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-log", type=int, default=10)
    parser.add_argument("--max-log", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--bandwidth-gbps", type=float, help="measured or vendor DRAM peak")
    parser.add_argument("--clmul-gops", type=float, help="measured clmul issue peak")
    parser.add_argument("--integer-gops", type=float, help="measured integer issue peak")
    parser.add_argument("--collapse-ratio", type=float, default=2.0)
    parser.add_argument("--loser-ratio", type=float, default=2.0)
    parser.add_argument("--op-counts", type=Path, help="JSON elements processed per prove")
    parser.add_argument("--json", type=Path, help="write machine-readable results")
    parser.add_argument(
        "--only",
        action="append",
        choices=[operation.name for operation in OPERATIONS],
        help="benchmark only this primitive (repeatable)",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    # Ring-switch transpose operates on 128-element rows.  Keeping the sweep
    # at or above that granularity lets every selected primitive share the
    # exact same element counts.
    if not 7 <= args.min_log <= args.max_log:
        raise SystemExit("expected 7 <= --min-log <= --max-log")
    if args.iterations < 1 or args.warmups < 0:
        raise SystemExit("--iterations must be positive and --warmups non-negative")
    peaks = (args.bandwidth_gbps, args.clmul_gops, args.integer_gops)
    if any(peak is not None and peak <= 0 for peak in peaks):
        raise SystemExit("roofline peaks must be positive")
    counts = _load_counts(args.op_counts)
    selected = [op for op in OPERATIONS if not args.only or op.name in args.only]
    print(f"device: {frx.devices()[0]} | backend: {frx.default_backend()}")
    results = []
    for operation in selected:
        print(f"sweeping {operation.name} ...", flush=True)
        results.append(
            run_operation(
                operation,
                range(args.min_log, args.max_log + 1),
                warmups=args.warmups,
                iterations=args.iterations,
                bandwidth_gbps=args.bandwidth_gbps,
                clmul_gops=args.clmul_gops,
                integer_gops=args.integer_gops,
                collapse_ratio=args.collapse_ratio,
                loser_ratio=args.loser_ratio,
                count_per_prove=counts.get(operation.name),
            )
        )
    _print_table(results)
    if args.json:
        payload = {
            "device": str(frx.devices()[0]),
            "backend": frx.default_backend(),
            "frx_version": getattr(frx, "__version__", "unknown"),
            "min_log": args.min_log,
            "max_log": args.max_log,
            "iterations": args.iterations,
            "warmups": args.warmups,
            "bandwidth_gbps": args.bandwidth_gbps,
            "clmul_gops": args.clmul_gops,
            "integer_gops": args.integer_gops,
            "collapse_ratio": args.collapse_ratio,
            "loser_ratio": args.loser_ratio,
            "results": [asdict(result) for result in results],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
