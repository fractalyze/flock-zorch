"""lincheck byte-match gate vs unmodified flock `lincheck::prove`.

Loads the golden (inputs + flock's LincheckProof), runs the frx port, and asserts
rounds + z_partial match byte-for-byte (identical under software and `clmad` mul).

Run:
  cargo run --release --example dump_lincheck -- 12 5 3 artifacts/lincheck_golden.bin
  JAX_PLATFORMS=cuda PYTHONPATH="python:$(scripts/zorch_pythonpath.sh)" <venv> \
      python/flock_zorch/lincheck/testing/lincheck_oracle_test.py
"""
import sys
from pathlib import Path

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)

from flock_zorch import ghash, lincheck  # noqa: E402

ART = Path(__file__).resolve().parents[4] / "artifacts"


class _Reader:
    def __init__(self, buf): self.b, self.o = buf, 0
    def u64(self):
        v = int.from_bytes(self.b[self.o:self.o + 8], "little"); self.o += 8; return v
    def f128(self):
        v = np.frombuffer(self.b, np.uint64, 2, self.o).copy(); self.o += 16; return v
    def f128s(self, n):
        v = np.frombuffer(self.b, np.uint64, 2 * n, self.o).reshape(n, 2).copy(); self.o += 16 * n; return v
    def raw(self, n):
        v = self.b[self.o:self.o + n]; self.o += n; return v


def _read_matrix(rd, k):
    dense = np.zeros((k, k), dtype=np.uint64)
    for r in range(k):
        nnz = rd.u64()
        for _ in range(nnz):
            dense[r, rd.u64()] = 1
    return dense


def _load(path):
    rd = _Reader(path.read_bytes())
    assert rd.raw(8) == b"FLKLIN01", "bad magic"
    m, k_log, k_skip = rd.u64(), rd.u64(), rd.u64()
    k = 1 << k_log
    a = _read_matrix(rd, k)
    b = _read_matrix(rd, k)
    z_packed = rd.raw(rd.u64())
    x_ab = lincheck.AbClaimPoint(z_skip=rd.f128(), x_inner_rest=rd.f128s(rd.u64()), x_outer=rd.f128s(rd.u64()))
    n_rounds = rd.u64()
    rounds = [(rd.f128(), rd.f128()) for _ in range(n_rounds)]
    z_partial = rd.f128s(rd.u64())
    return dict(m=m, k_log=k_log, k_skip=k_skip, a=a, b=b, z_packed=z_packed,
               x_ab=x_ab, rounds=rounds, z_partial=z_partial)


def _check(path, name):
    g = _load(path)
    lp = lincheck.prove(
        g["z_packed"], g["a"], g["b"], g["x_ab"], g["m"], g["k_log"], g["k_skip"])
    rounds, z_partial = lp.rounds, lp.z_partial
    assert len(rounds) == len(g["rounds"]), (len(rounds), len(g["rounds"]))
    for i, ((e1, einf), (ge1, geinf)) in enumerate(zip(rounds, g["rounds"])):
        assert np.array_equal(e1, ge1) and np.array_equal(einf, geinf), f"round {i} mismatch ({name})"
    assert np.array_equal(z_partial, g["z_partial"]), f"z_partial mismatch ({name})"
    return g["m"], g["k_log"], g["k_skip"]


def main() -> int:
    paths = sorted(ART.glob("lincheck_golden*.bin"))
    if not paths:
        print("no lincheck golden — run: cargo run --release --example dump_lincheck")
        return 1
    print(f"device: {frx.devices()[0]} | backend: {frx.default_backend()}")
    cfgs = [_check(p, "native") for p in paths]
    print(f"lincheck prove byte-match vs flock: PASS  cfgs(m,k_log,k_skip)={cfgs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
