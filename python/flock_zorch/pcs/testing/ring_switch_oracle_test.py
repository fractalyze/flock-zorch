"""Ring-switch byte-match gate vs flock `ring_switch::prove` (s_hat_v, rs_eq_ind, claim)."""
import sys
from pathlib import Path

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch import ghash  # noqa: E402
from flock_zorch.pcs import ring_switch  # noqa: E402
from flock_zorch.challenger import Challenger  # noqa: E402

ART = Path(__file__).resolve().parents[4] / "artifacts"


def _fv(raw, o):
    n = int.from_bytes(raw[o:o + 8], "little"); o += 8
    return np.frombuffer(raw, np.uint64, 2 * n, o).reshape(n, 2).copy(), o + 16 * n


def _check(name):
    raw = (ART / "ring_switch_golden.bin").read_bytes()
    assert raw[:8] == b"FLKRSW01"
    o = 16  # magic + m
    pw, o = _fv(raw, o)
    xo, o = _fv(raw, o)
    g_shv, o = _fv(raw, o)
    g_rei, o = _fv(raw, o)
    g_claim = np.frombuffer(raw, np.uint64, 2, o).copy()

    ch = Challenger(b"flock-ring-switch-test")
    shv, rei, claim = ring_switch.prove(pw, ghash.to_ghash(xo), ch)
    ok = (np.array_equal(ghash.to_lanes(shv), g_shv) and np.array_equal(ghash.to_lanes(rei), g_rei) and np.array_equal(ghash.to_lanes(claim), g_claim))
    bad = [k for k, v in {"s_hat_v": np.array_equal(ghash.to_lanes(shv), g_shv), "rs_eq_ind": np.array_equal(ghash.to_lanes(rei), g_rei),
                          "sumcheck_claim": np.array_equal(ghash.to_lanes(claim), g_claim)}.items() if not v]
    print(f"ring_switch byte-match vs flock ({name}): {'PASS' if ok else 'FAIL ' + str(bad)}")
    return ok


def main() -> int:
    print(f"device: {frx.devices()[0]} | backend: {frx.default_backend()}")
    ok = _check("software")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
