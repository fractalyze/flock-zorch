"""FS-wire gate for the zorch-driver Ligerito instantiation (flock-zorch#32 T4).

Two layers, both CPU, no golden:
  1. Framing lockstep: `FlockTranscript` / `FlockChoreography` byte streams equal
     the `Sha256Challenger` surface's (whose flock byte framing the proof-level oracle
     gates pin transitively), op by op — observes, scalar/slice sample split, PoW,
     and the rejection-sampled distinct query draw vs an independent Sha256Challenger
     reference.
  2. Round trip: `zorch.pcs.ligerito` prover+verifier over the GHASH
     `ReedSolomon` + the flock SHA-256 Merkle, driven end-to-end through the
     flock seams — verify ok, post-open == post-verify squeeze (FS lockstep),
     eager wire counts. The first zorch-driver run over the binary field; NOT
     yet a byte-match of flock's proof (the commit/induce basis convention is
     the remaining T4 delta, tracked on #32).
"""

import sys

import frx
import numpy as np

frx.config.update("jax_enable_x64", True)
frx.config.update("jax_platforms", "cpu")

import frx.numpy as fnp  # noqa: E402
from frx import lax  # noqa: E402
from zorch.coding.reed_solomon import ReedSolomon  # noqa: E402
from zorch.pcs.ligerito.config import LigeritoConfig  # noqa: E402
from zorch.pcs.ligerito.prover import LigeritoProver  # noqa: E402
from zorch.pcs.ligerito.verifier import LigeritoVerifier  # noqa: E402
from zorch.pcs.stage import OpeningClaim, OpeningWitness  # noqa: E402
from zorch.poly.multilinear import eval_mle  # noqa: E402

from flock_zorch.hash import merkle  # noqa: E402
from flock_zorch.pcs import ligerito as flock_ligerito  # noqa: E402
from flock_zorch.pcs.ligerito import (  # noqa: E402
    FlockChoreography,
    flock_ligerito_config,
    flock_transcript,
)
from flock_zorch.sha256_challenger import Sha256Challenger  # noqa: E402
from flock_zorch.testing._util import rand_ghash  # noqa: E402

DOMAIN = b"flock-ligerito-test"


def _ghash(lohi) -> fnp.ndarray:
    return lax.bitcast_convert_type(
        fnp.asarray(np.asarray(lohi, np.uint64)), fnp.binary_field_ghash
    )


def _lohi(x) -> np.ndarray:
    b = np.asarray(lax.bitcast_convert_type(x, fnp.uint8))
    return np.frombuffer(b.tobytes(), np.uint64).reshape(-1, 2)


def _state_eq(a, b) -> bool:
    """Streaming-transcript state equality — any framing divergence lands in the
    midstate/pending block, so this is the buffer-equality of the byte era."""
    sa, sb = a.state, b.state
    return all(
        np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in (
            (sa.h, sb.h),
            (sa.pending, sb.pending),
            (sa.pending_len, sb.pending_len),
            (sa.total_len, sb.total_len),
        )
    )


def test_wire_lohi_host_view():
    """The post-device_get wire view handles both scalar and vector GHASH."""
    lanes = np.array([[1, 2], [3, 4]], dtype=np.uint64)
    values = np.asarray(frx.device_get(_ghash(lanes)))
    check("wire lohi vector", np.array_equal(flock_ligerito._lohi(values), lanes))
    check(
        "wire lohi scalar",
        np.array_equal(flock_ligerito._lohi(values[0]), lanes[:1]),
    )


def test_packed_device_get_preserves_mixed_tree():
    """Packed D2H reconstruction preserves shape, dtype, and scalar leaves."""
    lanes = np.array([[1, 2], [3, 4]], dtype=np.uint64)
    tree = (
        _ghash(lanes),
        [
            _ghash(lanes[:1]).reshape(()),
            fnp.arange(5, dtype=fnp.int32),
            fnp.arange(7, dtype=fnp.uint8),
        ],
    )
    got = flock_ligerito._packed_device_get(tree)
    want = frx.device_get(tree)
    got_leaves = frx.tree_util.tree_leaves(got)
    want_leaves = frx.tree_util.tree_leaves(want)
    check(
        "packed device_get",
        len(got_leaves) == len(want_leaves)
        and all(
            x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)
            for x, y in zip(got_leaves, want_leaves)
        ),
    )


def check(name: str, ok: bool):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        sys.exit(1)


def test_observe_framing():
    """observe: ghash scalar == observe_f128, vector == per-element observe_f128,
    uint8 == observe_bytes — buffer-exact vs the Sha256Challenger-side ops."""
    vs = rand_ghash(np.random.default_rng(1), 3)
    root = np.arange(32, dtype=np.uint8)

    t = flock_transcript(DOMAIN)
    t = t.observe(vs[0]).observe(vs).observe(fnp.asarray(root))

    ch = Sha256Challenger(DOMAIN)
    ch.observe_f128(vs[0])
    for v in vs:
        ch.observe_f128(v)
    ch.observe_bytes(bytes(root))
    check("observe framing", _state_eq(t.inner, ch._t))


def test_sample_framing():
    """sample(1) == sample_f128 (scalar framing), sample(n) == sample_f128_vec
    (slice framing) — values and buffer."""
    t = flock_transcript(DOMAIN)
    t, one = t.sample(1)
    t, vec = t.sample(5)

    ch = Sha256Challenger(DOMAIN)
    ref_one = ch.sample_f128()
    ref_vec = ch.sample_f128(5)
    check(
        "sample values",
        np.array_equal(_lohi(one)[0], _lohi(ref_one)[0])
        and np.array_equal(_lohi(vec), _lohi(ref_vec)),
    )
    check("sample framing", _state_eq(t.inner, ch._t))


def test_grind_lockstep():
    """FlockChoreography grind/check == Sha256Challenger grind_pow / a fresh replay."""
    chor = FlockChoreography()
    t = flock_transcript(DOMAIN)
    t, w = chor.grind(t, 6)
    t, w0 = chor.grind(t, 0)  # flock's unconditional 0-bit query grind

    ch = Sha256Challenger(DOMAIN)
    n1, n0 = ch.grind_pow(6), ch.grind_pow(0)
    check("grind nonces", int(w) == n1 and int(w0) == n0 == 0)
    check("grind stream", _state_eq(t.inner, ch._t))

    v = flock_transcript(DOMAIN)
    v, ok1 = chor.check_grind(v, 6, w)
    v, ok0 = chor.check_grind(v, 0, w0)
    check("check_grind", bool(ok1) and bool(ok0) and _state_eq(v.inner, t.inner))


def _ref_distinct_queries(
    ch: Sha256Challenger, block_len: int, count: int
) -> list[int]:
    """flock's rejection-sampled distinct queries (sample an F128, take its low
    limb mod `block_len`, redraw on repeat, sort) — an independent Sha256Challenger-side
    reference for `FlockChoreography.sample_queries`, spelled out here so the gate
    holds without the retired in-tree Ligerito port."""
    seen: set[int] = set()
    out: list[int] = []
    while len(out) < count:
        q = int(_lohi(ch.sample_f128())[0, 0]) % block_len
        if q not in seen:
            seen.add(q)
            out.append(q)
    out.sort()
    return out


def test_distinct_queries_lockstep():
    """sample_queries == flock's distinct-query rejection sampling on equal states."""
    chor = FlockChoreography()
    t = flock_transcript(DOMAIN)
    t, pos = chor.sample_queries(t, block_len=16, count=6)

    ch = Sha256Challenger(DOMAIN)
    ref = _ref_distinct_queries(ch, 16, 6)
    check(
        "distinct queries",
        pos.tolist() == ref and _state_eq(t.inner, ch._t),
    )


def test_config_mapping():
    cfg = dict(
        initial_k=6,
        recursive_ks=[4, 3],
        log_inv_rates=[1, 2, 4],
        queries=[148, 100, 60],
        grinding_bits=[2, 1, 0],
        fold_grinding_bits=[3, 2, 0],
        ood_samples=[0, 1, 1],
        recursive_steps=2,
    )
    config, chor = flock_ligerito_config(cfg, log_n=15)
    check(
        "config mapping",
        config.fold_ks == (6, 4, 3)
        and config.ood_samples == (1, 1)
        and config.alpha_lsb_first
        and config.compressed_sumcheck_messages
        and config.monomial_commit
        and chor.fold_grinding_bits == (3, 2, 0)
        and chor.query_grinding_bits == (2, 1, 0),
    )


def test_round_trip_ghash():
    """zorch's ligerito over ReedSolomon(binary_field_ghash) + flock SHA-256
    Merkle, all FS through the flock seams: verify ok + FS lockstep."""
    config = LigeritoConfig(
        num_vars=6,
        fold_ks=(2, 2),
        log_inv_rates=(1, 2),
        queries=(4, 3),
        ood_samples=(1,),
        alpha_lsb_first=True,
        compressed_sumcheck_messages=True,
        monomial_commit=True,
    )
    chor = FlockChoreography(fold_grinding_bits=(1, 0), query_grinding_bits=(1, 0))

    def make_code(message_len: int, log_inv_rate: int) -> ReedSolomon:
        return ReedSolomon(
            message_len=message_len,
            blowup=1 << log_inv_rate,
            dtype=fnp.binary_field_ghash,
        )

    prover = LigeritoProver(make_code, merkle.GHASH_SHA256_TREE, config, chor)
    verifier = LigeritoVerifier(make_code, merkle.GHASH_SHA256_TREE, config, chor)

    f = rand_ghash(np.random.default_rng(7), 1 << config.num_vars)
    z = rand_ghash(np.random.default_rng(11), config.num_vars)
    root, pdata = prover.commit([f])
    claim = OpeningClaim(root, [z])
    opened = prover.prove(claim, OpeningWitness(pdata), flock_transcript(DOMAIN))
    value, proof = opened.reduction_proof.values, opened.reduction_proof.proof
    check("value = f(z)", np.array_equal(_lohi(value), _lohi(eval_mle(f, z))))
    check(
        "eager wire counts",
        len(proof.sumcheck_messages) == chor.num_messages(config)
        and len(proof.pow_witnesses) == chor.num_pow_witnesses(config),
    )
    verified = verifier.verify(claim, opened.reduction_proof, flock_transcript(DOMAIN))
    check("verify ok", bool(verified.ok))
    _, s_open = opened.transcript.sample(1)
    _, s_verify = verified.transcript.sample(1)
    check("FS lockstep", np.array_equal(_lohi(s_open), _lohi(s_verify)))


if __name__ == "__main__":
    test_wire_lohi_host_view()
    test_packed_device_get_preserves_mixed_tree()
    test_observe_framing()
    test_sample_framing()
    test_grind_lockstep()
    test_distinct_queries_lockstep()
    test_config_mapping()
    test_round_trip_ghash()
    print("OK zorch_ligerito_fs_test")
