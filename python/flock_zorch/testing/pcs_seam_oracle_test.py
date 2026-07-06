"""PcsProver-seam gate: `pcs.FlockPcsProver` commit/open byte-equal the raw
`pcs_commit.commit` + `pcs_open.open` on the pcs_open golden inputs (the raw
pair is golden-gated by pcs_open_oracle_test; the commit codeword is
additionally anchored to the golden directly).

Run:
  cargo run --release --example dump_pcs_open -- 20 1 2 artifacts/pcs_open_golden.bin
  bazel test //python:pcs_seam_oracle_test
"""
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from flock_zorch import pcs_commit, pcs_open  # noqa: E402
from flock_zorch.pcs import FlockPcsProver  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402

ART = Path(__file__).resolve().parents[3] / "artifacts"
DOMAIN = b"flock-pcs-seam-test"


class R:
    def __init__(s, b): s.b, s.o = b, 0
    def u(s): v = int.from_bytes(s.b[s.o:s.o + 8], "little"); s.o += 8; return v
    def fv(s): n = s.u(); v = np.frombuffer(s.b, np.uint64, 2 * n, s.o).reshape(n, 2).copy(); s.o += 16 * n; return v
    def raw(s, n): v = s.b[s.o:s.o + n]; s.o += n; return v


def _read_golden_inputs():
    rd = R((ART / "pcs_open_golden.bin").read_bytes())
    assert rd.raw(8) == b"FLKOPN01"
    m, lir, lbs = rd.u(), rd.u(), rd.u()
    return m, lir, lbs, rd.fv(), rd.fv(), rd.fv()  # z_packed, x_outer, codeword


def _deep_eq(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_deep_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return (isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(_deep_eq(x, y) for x, y in zip(a, b)))
    if isinstance(a, (int, bytes, np.integer)) and isinstance(b, (int, bytes, np.integer)):
        return a == b
    return np.array_equal(np.asarray(a), np.asarray(b))


def main() -> int:
    print(f"device: {jax.devices()[0]} | backend: {jax.default_backend()}")
    m, lir, lbs, z_packed, x_outer, g_codeword = _read_golden_inputs()
    k_code = (m - 7 - lbs) + lir

    root_r, cw_r, tree_r = pcs_commit.commit(z_packed, m, lir, lbs)
    ch_r = Challenger(DOMAIN)
    out_r = pcs_open.open(z_packed, cw_r, tree_r, x_outer, k_code, lir, lbs, ch_r)

    prover = FlockPcsProver(m, lir, lbs)
    root_s, data = prover.commit([z_packed])
    ch_s = Challenger(DOMAIN)
    values, proof, ch_out = prover.open(data, [x_outer], ch_s)

    checks = {
        "k_code": prover.k_code == k_code,
        "commit.root": np.array_equal(root_s, root_r),
        "commit.codeword_vs_golden": np.array_equal(data.codeword, g_codeword),
        "commit.tree": np.array_equal(data.tree, tree_r),
        "open.values_is_s_hat_v": np.array_equal(values, out_r["ring_switch"]),
        "open.proof": _deep_eq(proof, out_r),
        # Same object back + identical post-open FS state as the raw path.
        "open.transcript_threaded": ch_out is ch_s
                                    and np.array_equal(ch_out.sample_f128(), ch_r.sample_f128()),
    }
    ok = all(checks.values())
    bad = [k for k, v in checks.items() if not v]
    print(f"PcsProver seam equivalence (m={m} lir={lir} lbs={lbs}): "
          f"{'PASS' if ok else 'FAIL ' + str(bad)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
