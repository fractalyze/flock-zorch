"""Row-batch fold byte-match gate + FRI-arity/query KATs (PCS-open primitives)."""
import sys
from pathlib import Path

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)
import frx.numpy as jnp  # noqa: E402

from flock_zorch import ghash  # noqa: E402
from flock_zorch.pcs import fri  # noqa: E402

ART = Path(__file__).resolve().parents[4] / "artifacts"


def _kats() -> bool:
    ok = (fri.compute_fri_arities(17) == [6, 6, 5]
          and fri.compute_fri_arities(12) == [6, 6]
          and fri.compute_fri_arities(13) == [6, 6, 1]
          and fri.default_fri_queries(1) == 243
          and fri.default_fri_queries(2) == 148)
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
    got = np.asarray(frx.jit(lambda c, ch: fri.row_batch_fold_all(c, ch))(
        jnp.asarray(cw), jnp.asarray(chal)))
    ok = np.array_equal(got, folded)
    print(f"row_batch_fold byte-match vs flock (n_pos={n_pos} lbs={lbs}): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print(f"device: {frx.devices()[0]} | backend: {frx.default_backend()}")
    return 0 if (_kats() & _row_batch()) else 1


if __name__ == "__main__":
    sys.exit(main())
