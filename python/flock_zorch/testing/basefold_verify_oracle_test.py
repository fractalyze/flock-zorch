"""BaseFold verifier round-trip gate: `basefold.prove` output must VERIFY, and
tampered proofs must REJECT. Byte-anchored indirectly — the prover is byte-green
vs flock (`basefold_oracle_test`), so a verifier that accepts its output and
rejects tampering matches flock's `pcs::basefold::verify` acceptance semantics.

Runs over configs spanning 1/2/3 FRI epochs (arity schedule `compute_fri_arities`),
on CPU (bazel `//python:all`) or GPU (venv, `JAX_PLATFORMS=cuda`).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from flock_zorch import field, basefold, merkle, pcs_commit, fri  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402

DOMAIN = b"flock-basefold-test"
ART = Path(__file__).resolve().parents[3] / "artifacts"


class _R:
    """Reader for the flock-dumped `basefold_golden.bin` (matches the layout in
    `basefold_oracle_test`)."""
    def __init__(s, b): s.b, s.o = b, 0
    def u(s): v = int.from_bytes(s.b[s.o:s.o + 8], "little"); s.o += 8; return v
    def f(s): v = np.frombuffer(s.b, np.uint64, 2, s.o).copy(); s.o += 16; return v
    def fv(s): n = s.u(); v = np.frombuffer(s.b, np.uint64, 2 * n, s.o).reshape(n, 2).copy(); s.o += 16 * n; return v
    def h(s): v = np.frombuffer(s.b, np.uint8, 32, s.o).copy(); s.o += 32; return v
    def hv(s): n = s.u(); v = np.frombuffer(s.b, np.uint8, 32 * n, s.o).reshape(n, 32).copy(); s.o += 32 * n; return v
    def raw(s, n): v = s.b[s.o:s.o + n]; s.o += n; return v


def _check_golden(path):
    """Byte-anchor: `basefold.verify` must ACCEPT a proof produced by UNMODIFIED
    flock, replaying the same challenger domain the fixture was dumped with.
    Stronger than the python round-trip — the proof bytes are flock's own. The
    config (m/lir/lbs) is read from the fixture header, so one reader drives
    whatever epoch schedule the golden was dumped at. Missing golden is a hard
    error (matches `basefold_oracle_test`); the gate never degrades to a vacuous
    self-consistent round-trip."""
    rd = _R(path.read_bytes())
    assert rd.raw(8) == b"FLKBSF01"
    m, lir, lbs, n_q = rd.u(), rd.u(), rd.u(), rd.u()
    z_packed, b, codeword = rd.fv(), rd.fv(), rd.fv()
    n_rm = rd.u(); rm = [(rd.f(), rd.f()) for _ in range(n_rm)]
    post_root = rd.h(); rc = rd.hv()
    fa, fb, fcw = rd.f(), rd.f(), rd.fv()
    nq = rd.u(); queries = []
    for _ in range(nq):
        pos = rd.u(); il = rd.fv(); prl = rd.fv(); ne = rd.u(); el = [rd.fv() for _ in range(ne)]
        queries.append((pos, il, prl, el))
    imp, prmp = rd.hv(), rd.hv()
    n_emp = rd.u(); emp = [rd.hv() for _ in range(n_emp)]

    proof = {
        "round_messages": rm, "post_row_batch_commit": post_root,
        "round_commitments": list(rc), "final_a": fa, "final_b": fb,
        "final_codeword": fcw, "queries": queries, "initial_multi_proof": imp,
        "post_row_batch_multi_proof": prmp, "epoch_multi_proofs": emp,
    }
    k_code = (m - field.LOG_PACKING - lbs) + lir
    num_ntts = 1 << lbs
    n_leaves = 1 << k_code
    tree = merkle.merkle_tree(codeword.reshape(n_leaves, num_ntts * 2).view(np.uint8))
    root = tree[-1]
    target = np.asarray(field.sum(field.mul(jnp.asarray(z_packed), jnp.asarray(b))))
    ch = Challenger(DOMAIN)
    ok, info = basefold.verify(target, proof, root, k_code, lir, lbs, ch)
    arities = fri.compute_fri_arities(k_code - lir)
    print(f"golden accept (flock's own proof, m={m} lir={lir} lbs={lbs}, "
          f"{len(arities)} epoch, arities={arities}): "
          f"{'PASS' if ok else 'FAIL ' + info['reason']}")
    return ok


def _build(m, lir, lbs, seed):
    """Prove a random claim; return (proof, target, root, k_code)."""
    LOGP = field.LOG_PACKING
    k_code = (m - LOGP - lbs) + lir
    n_a = 1 << (m - LOGP)
    n_q = fri.default_fri_queries(lir)
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2**64, size=(n_a, 2), dtype=np.uint64)
    b = rng.integers(0, 2**64, size=(n_a, 2), dtype=np.uint64)
    root, codeword, tree = pcs_commit.commit(z, m, lir, lbs)
    target = np.asarray(field.sum(field.mul(jnp.asarray(z), jnp.asarray(b))))
    ch = Challenger(DOMAIN)
    proof = basefold.prove(z, b, codeword, tree, k_code, lir, lbs, n_q, ch)
    return proof, target, root, k_code


def _verify(proof, target, root, k_code, lir, lbs):
    ch = Challenger(DOMAIN)
    return basefold.verify(target, proof, root, k_code, lir, lbs, ch)


def _check_accept(m, lir, lbs, seed):
    proof, target, root, k_code = _build(m, lir, lbs, seed)
    ok, info = _verify(proof, target, root, k_code, lir, lbs)
    arities = fri.compute_fri_arities(k_code - lir)
    print(f"accept (m={m} lir={lir} lbs={lbs}, {len(arities)} epoch, arities={arities}): "
          f"{'PASS' if ok else 'FAIL ' + info['reason']}")
    return ok, (proof, target, root, k_code, lir, lbs)


def _check_reject(name, mutate, base):
    proof, target, root, k_code, lir, lbs = base
    p2 = copy.deepcopy(proof)
    t2, r2 = target, root
    args = mutate(p2, target, root)
    if args is not None:
        p2, t2, r2 = args
    ok, info = _verify(p2, t2, r2, k_code, lir, lbs)
    print(f"  reject[{name}]: {'PASS (rejected: ' + info['reason'] + ')' if not ok else 'FAIL (accepted!)'}")
    return not ok


def main() -> int:
    print(f"device: {jax.devices()[0]} | backend: {jax.default_backend()}")
    ok = True

    # Byte-anchor: accept unmodified flock's own proof, 1-epoch and multi-epoch.
    ok = _check_golden(ART / "basefold_golden.bin") and ok         # 1 epoch  [5]
    ok = _check_golden(ART / "basefold_3epoch_golden.bin") and ok  # 3 epochs [6,6,1]

    # Python round-trip across the FRI-epoch schedule.
    a1, _ = _check_accept(14, 1, 2, 0); ok = a1 and ok       # 1 epoch  [5]
    a2, base2 = _check_accept(16, 1, 2, 1); ok = a2 and ok    # 2 epochs [6,1]
    a3, _ = _check_accept(22, 1, 2, 2); ok = a3 and ok        # 3 epochs [6,6,1]
    a4, _ = _check_accept(16, 1, 1, 3); ok = a4 and ok        # lbs=1    [6,2]

    # Tamper vectors on the 2-epoch proof.
    print("tamper (2-epoch base):")

    def t_msg(p, t, r):
        p["round_messages"][0] = (p["round_messages"][0][0] ^ np.uint64(1), p["round_messages"][0][1])
    def t_leaf(p, t, r):
        q0 = list(p["queries"][0]); leaf = q0[1].copy(); leaf[0, 0] ^= np.uint64(1); q0[1] = leaf
        p["queries"][0] = tuple(q0)
    def t_final(p, t, r):
        p["final_a"] = p["final_a"] ^ np.uint64(1)
    def t_root(p, t, r):
        return p, t, (np.asarray(r).copy() ^ np.uint8(1))

    ok = _check_reject("round_message", t_msg, base2) and ok
    ok = _check_reject("query_leaf", t_leaf, base2) and ok
    ok = _check_reject("final_a", t_final, base2) and ok
    ok = _check_reject("initial_root", t_root, base2) and ok

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
