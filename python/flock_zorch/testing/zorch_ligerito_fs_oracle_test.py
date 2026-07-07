"""FS-wire gate for the zorch-driver Ligerito instantiation (flock-zorch#32 T4).

Two layers, both CPU:
  1. Framing lockstep: `FlockTranscript` / `FlockChoreography` byte streams equal
     the `Challenger` surface's (which `challenger_oracle_test` pins to
     flock-core), op by op — observes, scalar/slice sample split, PoW, and the
     rejection-sampled distinct query draw vs an independent Challenger reference.
  2. Round trip: `zorch.pcs.ligerito` prover+verifier over the GHASH
     `ReedSolomon` + the flock SHA-256 Merkle, driven end-to-end through the
     flock seams — verify ok, post-open == post-verify squeeze (FS lockstep),
     eager wire counts. The first zorch-driver run over the binary field; NOT
     yet a byte-match of flock's proof (the commit/induce basis convention is
     the remaining T4 delta, tracked on #32).
"""
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from zorch.byte_transcript import ByteHashTranscript  # noqa: E402
from zorch.coding.reed_solomon import ReedSolomon  # noqa: E402
from zorch.hash.sha256 import HashlibSha256  # noqa: E402
from zorch.pcs.ligerito.config import LigeritoConfig  # noqa: E402
from zorch.pcs.ligerito.prover import LigeritoProver  # noqa: E402
from zorch.pcs.ligerito.verifier import LigeritoVerifier  # noqa: E402
from zorch.poly.multilinear import eval_mle  # noqa: E402

from flock_zorch import merkle  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.zorch_ligerito import (  # noqa: E402
    FlockChoreography,
    FlockTranscript,
    flock_ligerito_config,
    flock_transcript,
)

DOMAIN = b"flock-ligerito-test"


def _ghash(lohi) -> jnp.ndarray:
    return lax.bitcast_convert_type(
        jnp.asarray(np.asarray(lohi, np.uint64)), jnp.binary_field_ghash
    )


def _rand_ghash(seed: int, n: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return _ghash(rng.integers(0, 1 << 63, size=(n, 2), dtype=np.uint64))


def _lohi(x) -> np.ndarray:
    b = np.asarray(lax.bitcast_convert_type(x, jnp.uint8))
    return np.frombuffer(b.tobytes(), np.uint64).reshape(-1, 2)


def check(name: str, ok: bool):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        sys.exit(1)


def test_observe_framing():
    """observe: ghash scalar == observe_f128, vector == per-element observe_f128,
    uint8 == observe_bytes — buffer-exact vs the Challenger-side ops."""
    vs = _rand_ghash(1, 3)
    root = np.arange(32, dtype=np.uint8)

    t = flock_transcript(DOMAIN)
    t = t.observe(vs[0]).observe(vs).observe(jnp.asarray(root))

    ch = Challenger(DOMAIN)
    ch.observe_f128(_lohi(vs[0])[0])
    for v in _lohi(vs):
        ch.observe_f128(v)
    ch.observe_bytes(bytes(root))
    check("observe framing", t.inner.buffer == ch._t.buffer)


def test_sample_framing():
    """sample(1) == sample_f128 (scalar framing), sample(n) == sample_f128_vec
    (slice framing) — values and buffer."""
    t = flock_transcript(DOMAIN)
    t, one = t.sample(1)
    t, vec = t.sample(5)

    ch = Challenger(DOMAIN)
    ref_one = ch.sample_f128()
    ref_vec = ch.sample_f128_vec(5)
    check(
        "sample values",
        np.array_equal(_lohi(one)[0], ref_one)
        and np.array_equal(_lohi(vec), ref_vec),
    )
    check("sample framing", t.inner.buffer == ch._t.buffer)


def test_grind_lockstep():
    """FlockChoreography grind/check == Challenger grind_pow/verify_pow."""
    chor = FlockChoreography()
    t = flock_transcript(DOMAIN)
    t, w = chor.grind(t, 6)
    t, w0 = chor.grind(t, 0)  # flock's unconditional 0-bit query grind

    ch = Challenger(DOMAIN)
    n1, n0 = ch.grind_pow(6), ch.grind_pow(0)
    check("grind nonces", int(w) == n1 and int(w0) == n0 == 0)
    check("grind stream", t.inner.buffer == ch._t.buffer)

    v = flock_transcript(DOMAIN)
    v, ok1 = chor.check_grind(v, 6, w)
    v, ok0 = chor.check_grind(v, 0, w0)
    check("check_grind", bool(ok1) and bool(ok0) and v.inner.buffer == t.inner.buffer)


def _ref_distinct_queries(ch: Challenger, block_len: int, count: int) -> list[int]:
    """flock's rejection-sampled distinct queries (sample an F128, take its low
    limb mod `block_len`, redraw on repeat, sort) — an independent Challenger-side
    reference for `FlockChoreography.sample_queries`, spelled out here so the gate
    holds without the retired in-tree Ligerito port."""
    seen: set[int] = set()
    out: list[int] = []
    while len(out) < count:
        q = int(ch.sample_f128()[0]) % block_len
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

    ch = Challenger(DOMAIN)
    ref = _ref_distinct_queries(ch, 16, 6)
    check(
        "distinct queries",
        pos.tolist() == ref and t.inner.buffer == ch._t.buffer,
    )


def test_config_mapping():
    cfg = dict(
        initial_k=6, recursive_ks=[4, 3], log_inv_rates=[1, 2, 4],
        queries=[148, 100, 60], grinding_bits=[2, 1, 0],
        fold_grinding_bits=[3, 2, 0], ood_samples=[0, 1, 1], recursive_steps=2,
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
            dtype=jnp.binary_field_ghash,
        )

    prover = LigeritoProver(make_code, merkle.GHASH_TREE, config, chor)
    verifier = LigeritoVerifier(make_code, merkle.GHASH_TREE, config, chor)

    f = _rand_ghash(7, 1 << config.num_vars)
    z = _rand_ghash(11, config.num_vars)
    root, pdata = prover.commit([f])
    value, proof, t_open = prover.open(pdata, [z], flock_transcript(DOMAIN))
    check("value = f(z)", np.array_equal(_lohi(value), _lohi(eval_mle(f, z))))
    check(
        "eager wire counts",
        len(proof.sumcheck_messages) == chor.num_messages(config)
        and len(proof.pow_witnesses) == chor.num_pow_witnesses(config),
    )
    ok, t_verify = verifier.verify(root, [z], value, proof, flock_transcript(DOMAIN))
    check("verify ok", bool(ok))
    _, s_open = t_open.sample(1)
    _, s_verify = t_verify.sample(1)
    check("FS lockstep", np.array_equal(_lohi(s_open), _lohi(s_verify)))


if __name__ == "__main__":
    test_observe_framing()
    test_sample_framing()
    test_grind_lockstep()
    test_distinct_queries_lockstep()
    test_config_mapping()
    test_round_trip_ghash()
    print("OK zorch_ligerito_fs_oracle_test")
