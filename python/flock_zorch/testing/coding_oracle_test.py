"""Additive-RS FoldableCode gate — zorch `coding.AdditiveReedSolomon` vs flock's
FRI fold family, ahead of `basefold.py` adopting `code.fold` (issue #12 task 2).

Three anchors: (1) the device twiddle table byte-matches flock's host
`ntt.compute_twiddles` at the golden's k_code; (2) `code.fold` reproduces
`fri_fold_golden.bin` byte-for-byte through the SoA↔ghash boundary; (3)
`code.fold` == `fri.fri_fold` and `code.fold_values` == gathered `code.fold`
at EVERY layer of a seeded fold chain, not just the golden's one layer.

Run:
  bazel test //python:coding_oracle_test
"""
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from zorch.coding.additive_reed_solomon import (  # noqa: E402
    AdditiveReedSolomon,
    additive_ntt_twiddles,
)

from flock_zorch import fri, ntt as ntt_mod  # noqa: E402

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


def _ghash(soa):
    """uint64 [..., 2] SoA -> binary_field_ghash [...] (device bitcast)."""
    return lax.bitcast_convert_type(jnp.asarray(soa), jnp.binary_field_ghash)


def _soa(g):
    """ghash [n] -> uint64 [n, 2] via host bytes (the device ghash->uint64
    bitcast silently returns zeros, zorch#399)."""
    arr = np.asarray(g)
    return np.frombuffer(arr.tobytes(), np.uint64).reshape(arr.shape[0], 2)


def main() -> int:
    k_code, layer, chal, cw, golden = _load()
    print(f"device: {jax.devices()[0]} | backend: {jax.default_backend()}")

    ok = np.array_equal(
        _soa(additive_ntt_twiddles(k_code, jnp.binary_field_ghash)),
        np.asarray(ntt_mod.compute_twiddles(k_code)),
    )
    print(f"twiddle table byte-match vs ntt.compute_twiddles (k_code={k_code}): "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        return 1

    # (2,)-uint64 bitcasts to a 0-d ghash scalar; keep the challenge (1,).
    code = AdditiveReedSolomon(1 << (k_code - 1), 2, jnp.binary_field_ghash)
    got = _soa(jax.jit(code.fold)(_ghash(cw), _ghash(chal.reshape(1, 2))))
    ok = np.array_equal(got, golden)
    print(f"code.fold byte-match vs flock golden (k_code={k_code} layer={layer}): "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        return 1

    # Full fold chain, each side advancing on its own output; equality per layer.
    k = 12
    rng = np.random.default_rng(0)
    code = AdditiveReedSolomon(1 << (k - 1), 2, jnp.binary_field_ghash)
    tw = jnp.asarray(ntt_mod.compute_twiddles(k))
    cur_soa = jnp.asarray(rng.integers(0, 1 << 64, size=(1 << k, 2), dtype=np.uint64))
    cur_g = _ghash(cur_soa)
    for level in range(k):
        ch = rng.integers(0, 1 << 64, size=(1, 2), dtype=np.uint64)
        beta = _ghash(ch)
        ref = fri.fri_fold(cur_soa, tw, k - level - 1, jnp.asarray(ch[0]))
        folded_g = code.fold(cur_g, beta)
        if not np.array_equal(_soa(folded_g), np.asarray(ref)):
            print(f"code.fold differential vs fri.fri_fold: FAIL at level {level}")
            return 1

        half = 1 << (k - level - 1)
        positions = jnp.asarray(rng.integers(0, half, size=min(8, half)))
        fv = code.fold_values(cur_g[2 * positions], cur_g[2 * positions + 1],
                              beta, positions, level)
        if not np.array_equal(_soa(fv), _soa(folded_g[positions])):
            print(f"code.fold_values vs gathered code.fold: FAIL at level {level}")
            return 1
        cur_soa, cur_g = ref, folded_g
    print(f"code.fold + fold_values differential vs fri.fri_fold ({k} layers): PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
