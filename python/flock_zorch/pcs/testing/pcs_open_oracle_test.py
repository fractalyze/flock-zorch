"""Full single-claim PCS open byte-match gate vs flock `pcs::open`.

The capstone of the PCS-open layer: ring-switch + BaseFold on one challenger.
Compares s_hat_v + every BaseFoldProof field byte-for-byte (software + `clmad` mul).

Run:
  cargo run --release --example dump_pcs_open -- 20 1 2 artifacts/pcs_open_golden.bin
  JAX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/pcs/testing/pcs_open_oracle_test.py
"""
import sys
from pathlib import Path

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch import ghash  # noqa: E402
from flock_zorch.pcs import open as pcs_open  # noqa: E402
from flock_zorch.hash import merkle  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402

ART = Path(__file__).resolve().parents[4] / "artifacts"


class R:
    def __init__(s, b): s.b, s.o = b, 0
    def u(s): v = int.from_bytes(s.b[s.o:s.o + 8], "little"); s.o += 8; return v
    def f(s): v = np.frombuffer(s.b, np.uint64, 2, s.o).copy(); s.o += 16; return v
    def fv(s): n = s.u(); v = np.frombuffer(s.b, np.uint64, 2 * n, s.o).reshape(n, 2).copy(); s.o += 16 * n; return v
    def h(s): v = np.frombuffer(s.b, np.uint8, 32, s.o).copy(); s.o += 32; return v
    def hv(s): n = s.u(); v = np.frombuffer(s.b, np.uint8, 32 * n, s.o).reshape(n, 32).copy(); s.o += 32 * n; return v
    def raw(s, n): v = s.b[s.o:s.o + n]; s.o += n; return v


def _check(name):
    rd = R((ART / "pcs_open_golden.bin").read_bytes())
    assert rd.raw(8) == b"FLKOPN01"
    m, lir, lbs = rd.u(), rd.u(), rd.u()
    z_packed, x_outer, codeword = rd.fv(), rd.fv(), rd.fv()
    g_shv = rd.fv()
    n_rm = rd.u(); g_rm = [(rd.f(), rd.f()) for _ in range(n_rm)]
    g_post = rd.h(); g_rc = rd.hv()
    g_fa, g_fb, g_fcw = rd.f(), rd.f(), rd.fv()
    n_q = rd.u(); g_q = []
    for _ in range(n_q):
        pos = rd.u(); il = rd.fv(); prl = rd.fv(); ne = rd.u(); el = [rd.fv() for _ in range(ne)]
        g_q.append((pos, il, prl, el))
    g_imp, g_prmp = rd.hv(), rd.hv()
    n_emp = rd.u(); g_emp = [rd.hv() for _ in range(n_emp)]

    k_code = (m - 7 - lbs) + lir
    num_ntts = 1 << lbs
    init_tree = merkle.merkle_tree(codeword.reshape(1 << k_code, num_ntts * 2).view(np.uint8))
    ch = Challenger(b"flock-pcs-open-test")
    out = pcs_open.open(z_packed, codeword, init_tree, x_outer, k_code, lir, lbs, ch)
    p = out.basefold

    def eq(x, y): return np.array_equal(np.asarray(x), np.asarray(y))
    checks = {
        "ring_switch.s_hat_v": eq(out.ring_switch, g_shv),
        "round_messages": all(eq(a, c) and eq(b_, d) for (a, b_), (c, d) in zip(p.round_messages, g_rm)),
        "post_row_batch_commit": eq(p.post_row_batch_commit, g_post),
        "round_commitments": eq(np.stack(p.round_commitments) if p.round_commitments else np.zeros((0, 32), np.uint8), g_rc),
        "final_a": eq(p.final_a, g_fa), "final_b": eq(p.final_b, g_fb), "final_codeword": eq(p.final_codeword, g_fcw),
        "queries": all(q[0] == gq[0] and eq(q[1], gq[1]) and eq(q[2], gq[2]) and all(eq(x, y) for x, y in zip(q[3], gq[3]))
                       for q, gq in zip(p.queries, g_q)),
        "initial_multi_proof": eq(p.initial_multi_proof, g_imp),
        "post_row_batch_multi_proof": eq(p.post_row_batch_multi_proof, g_prmp),
        "epoch_multi_proofs": all(eq(a, b_) for a, b_ in zip(p.epoch_multi_proofs, g_emp)),
    }
    ok = all(checks.values())
    bad = [k for k, v in checks.items() if not v]
    print(f"pcs::open byte-match vs flock ({name}, m={m} lir={lir} lbs={lbs}): "
          f"{'PASS' if ok else 'FAIL ' + str(bad)}")
    return ok


def main() -> int:
    print(f"device: {frx.devices()[0]} | backend: {frx.default_backend()}")
    ok = _check("software")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
