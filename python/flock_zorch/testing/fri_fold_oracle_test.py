"""FRI codeword-fold byte-match gate + GPU-vs-CPU speed (PCS-open compute).

Loads flock's golden (a fold of a 2^(layer+1) codeword) and asserts the jax port
reproduces it byte-for-byte (software + clmad), then times the GPU fold vs the CPU
fold (the open's dominant compute is log_dim of these).

Run:
  cargo run --release --example dump_fri_fold -- 20 19 artifacts/fri_fold_golden.bin
  JAX_PLATFORMS=cuda PYTHONPATH=python:/home/jooman/fractalyze/zorch <venv> \
      python/flock_zorch/testing/fri_fold_oracle_test.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flock_zorch import field, ntt as ntt_mod, pcs_open  # noqa: E402

ART = Path(__file__).resolve().parents[3] / "artifacts"


def _load():
    raw = (ART / "fri_fold_golden.bin").read_bytes()
    assert raw[:8] == b"FLKFRI01", "bad magic"
    k_code = int.from_bytes(raw[8:16], "little")
    layer = int.from_bytes(raw[16:24], "little")
    off = 24
    chal = np.frombuffer(raw, np.uint64, 2, off); off += 16
    n = 1 << (layer + 1)
    cw = np.frombuffer(raw, np.uint64, n * 2, off).reshape(n, 2); off += n * 16
    folded = np.frombuffer(raw, np.uint64, (n // 2) * 2, off).reshape(n // 2, 2)
    return k_code, layer, chal, cw, folded


def main() -> int:
    k_code, layer, chal, cw, golden = _load()
    print(f"device: {jax.devices()[0]} | backend: {jax.default_backend()}")
    tw = jnp.asarray(ntt_mod.compute_twiddles(k_code))
    cwj, chj = jnp.asarray(cw), jnp.asarray(chal)

    muls = [("software", field.mul)]
    try:
        from flock_zorch import field_clmad
        if field_clmad.available():
            muls.append(("clmad", field_clmad.mul))
    except Exception:  # noqa: BLE001
        pass
    for name, mul in muls:
        got = np.asarray(jax.jit(lambda c, t, ch: pcs_open.fri_fold(c, t, layer, ch, mul=mul))(cwj, tw, chj))
        ok = np.array_equal(got, golden)
        print(f"FRI fold byte-match vs flock ({name}, k_code={k_code} layer={layer}): "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
