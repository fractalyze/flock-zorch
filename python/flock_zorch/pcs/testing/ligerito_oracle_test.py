"""Ligerito recursive PCS byte gate vs flock — the CPU-CI byte anchor.

Ingests the dump_ligerito golden (config + synthetic f/b/target + L0 commit + full
LigeritoProof) and byte-compares the whole flock `LigeritoProof` — every field,
including the octopus `merkle_proof` — produced by `zorch_ligerito.prove_flock_ligerito`
(the flock-zorch prove path: `zorch.pcs.ligerito` driven through the flock FS seam,
octopus reassembled from the zorch openings). The only per-layer golden gate kept
after the proof-level cutover: the fused e2e/hash-circuit gates need a GPU, and
flock's fused prove has no registered Ligerito config below m=22, so this is the
one byte-match `bazel test` can run on CPU CI.

Run under bazel (bazel test //python:ligerito_oracle_test) or on the venv
(regen golden: cargo run --release --example dump_ligerito -- 15 artifacts/ligerito_golden.bin):
  FRX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/pcs/testing/ligerito_oracle_test.py
"""
import sys

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch import ghash  # noqa: E402
from flock_zorch.pcs import ligerito as zorch_ligerito  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402
from flock_zorch.testing._golden import (  # noqa: E402
    ligerito_proof_results, open_golden, read_ligerito_config)
from flock_zorch.testing._util import report  # noqa: E402


def load():
    rd = open_golden("ligerito_golden.bin")
    assert bytes(rd.take(8)) == b"FLKLIG01", "bad magic"
    g = dict(log_n=rd.u(), m=rd.u(), lbs=rd.u())
    g["cfg"] = read_ligerito_config(rd)
    g["f"] = rd.fv(); g["b"] = rd.fv(); g["target"] = rd.f()
    g["l0_codeword"] = rd.fv(); g["l0_tree"] = rd.hv()
    g["initial_root"] = rd.raw(32)
    # LigeritoProof
    g["initial_proof"] = dict(opened_rows=rd.rows(), merkle_proof=rd.hv())
    g["recursive_roots"] = rd.hv()
    nrp = rd.u(); g["recursive_proofs"] = [dict(opened_rows=rd.rows(), merkle_proof=rd.hv()) for _ in range(nrp)]
    g["final_proof"] = dict(yr=rd.fv(), opened_rows=rd.rows(), merkle_proof=rd.hv())
    g["sumcheck_transcript"] = rd.pair()
    g["grinding_nonces"] = rd.u64v()
    g["ood_values"] = rd.fv()
    g["fold_grinding_nonces"] = rd.u64v()
    return rd, g


def run():
    _, g = load()
    cfg = g["cfg"]
    results = []

    # The flock-zorch prove path → byte-gate every LigeritoProof field.
    ch = Challenger(b"flock-ligerito-test")
    _root, pdata = zorch_ligerito.commit_flock_ligerito(cfg, g["f"])
    p = zorch_ligerito.prove_flock_ligerito(cfg, pdata, ghash.to_ghash(g["b"]),
                                            ghash.to_ghash(g["target"]), ch)

    results.extend(ligerito_proof_results(p, g, prefix=""))
    return g, results


def main() -> int:
    print(f"device {frx.devices()[0]}")
    g, results = run()
    return report(results, f"ligerito prove_flock_ligerito vs flock (log_n={g['log_n']}, "
                           f"R={g['cfg']['recursive_steps']})")


if __name__ == "__main__":
    sys.exit(main())
