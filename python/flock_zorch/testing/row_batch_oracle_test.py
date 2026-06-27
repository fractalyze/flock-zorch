"""Row-batch fold byte-match gate + FRI-arity/query KATs (PCS-open primitives)."""
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flock_zorch import field, pcs_open  # noqa: E402

ART = Path(__file__).resolve().parents[3] / "artifacts"


def _kats() -> bool:
    ok = (pcs_open.compute_fri_arities(17) == [6, 6, 5]
          and pcs_open.compute_fri_arities(12) == [6, 6]
          and pcs_open.compute_fri_arities(13) == [6, 6, 1]
          and pcs_open.default_fri_queries(1) == 243
          and pcs_open.default_fri_queries(2) == 148)
    print(f"FRI arities/queries KAT: {'PASS' if ok else 'FAIL'}")
    return ok


def _row_batch() -> bool:
    raw = (ART / "row_batch_golden.bin").read_bytes()
    assert raw[:8] == b"FLKRBF01", "bad magic"
    n_pos = int.from_bytes(raw[8:16], "little")
    lbs = int.from_bytes(raw[16:24], "little")
    num_ntts = 1 << lbs
    off = 24
    chal = np.frombuffer(raw, np.uint64, lbs * 2, off).reshape(lbs, 2); off += lbs * 16
    cw = np.frombuffer(raw, np.uint64, n_pos * num_ntts * 2, off).reshape(n_pos * num_ntts, 2)
    off += n_pos * num_ntts * 16
    folded = np.frombuffer(raw, np.uint64, n_pos * 2, off).reshape(n_pos, 2)
    ok = True
    for name, mul in (("software", field.mul),):
        got = np.asarray(jax.jit(lambda c, ch: pcs_open.row_batch_fold_all(c, ch, mul=mul))(
            jnp.asarray(cw), jnp.asarray(chal)))
        good = np.array_equal(got, folded)
        print(f"row_batch_fold byte-match vs flock ({name}, n_pos={n_pos} lbs={lbs}): "
              f"{'PASS' if good else 'FAIL'}")
        ok = ok and good
    return ok


def main() -> int:
    print(f"device: {jax.devices()[0]} | backend: {jax.default_backend()}")
    return 0 if (_kats() & _row_batch()) else 1


if __name__ == "__main__":
    sys.exit(main())
