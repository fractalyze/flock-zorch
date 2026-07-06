"""M0 byte gate: zorch's Reed-Solomon reproduces flock's L0 Ligero commitment.

Loads the `dump_ligerito` golden and checks that zorch's binary-field additive-NTT
`ReedSolomon.encode` (via `flock_zorch.zorch_ligero`) produces flock-core's L0
codeword + Merkle root byte-for-byte — the commitment-level de-risk of
instantiating zorch's code-generic Ligerito over GHASH (flock-zorch#32, T4a).

This is the first flock-zorch gate to consume zorch's PCS layer. It reuses the
golden reader from `ligerito_oracle_test` (same fixture, same config), so a green
run here plus a green M0 there transitively pins zorch's encode to flock-core.

GPU only — binary-field arithmetic is unlowered on this env's CPU PJRT path.

Run (regen golden: cargo run --release --example dump_ligerito -- 15 artifacts/ligerito_golden.bin):
  export PATH="$HOME/.local/cuda13/bin:$PATH"
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
    PYTHONPATH=python:/home/ryan/Workspace/envs/flock-zorch2/zorch \
    .venv/bin/python python/flock_zorch/testing/zorch_ligero_commit_oracle_test.py
"""

import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from flock_zorch import zorch_ligero  # noqa: E402
from flock_zorch.testing import ligerito_oracle_test as ot  # noqa: E402


def run():
    _, g = ot.load()
    cfg = g["cfg"]
    codeword, tree = zorch_ligero.ligero_commit(
        zorch_ligero.to_ghash(g["f"]),
        cfg["initial_log_msg_cols"],
        cfg["initial_log_num_interleaved"],
        cfg["log_inv_rates"][0],
    )
    return g, [
        ("L0 codeword == flock l0_codeword",
         np.array_equal(codeword, g["l0_codeword"])),
        ("L0 root == flock initial_root",
         np.array_equal(tree[-1], g["initial_root"])),
    ]


def main() -> int:
    print(f"device {jax.devices()[0]}")
    g, results = run()
    allok = True
    for nm, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}")
        allok = allok and ok
    print(f"zorch-backed Ligero L0 commit vs flock-core "
          f"(log_n={g['log_n']}): {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
