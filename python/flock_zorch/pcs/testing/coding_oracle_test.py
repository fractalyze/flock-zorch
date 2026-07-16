"""Additive-RS FoldableCode gate — zorch `coding.AdditiveReedSolomon` byte-equals
flock's additive-NTT encode + FRI fold, anchored to the flock-core `.bin`
goldens (issue #12 task 2). Consolidates the old `ntt_oracle_test` (encode) and
`fri_fold_oracle_test` (fold) onto the code the prover actually runs.

Anchors: (1) the device twiddle table byte-matches `ntt_golden.bin`'s table;
(2) `code.encode` reproduces the NTT golden's output through the SoA↔ghash
boundary; (3) `code.fold` reproduces `fri_fold_golden.bin`; (4) `code.fold_values`
agrees with `code.fold` on the golden codeword. The multi-layer fold chain +
final-codeword equality are covered against the `fri_fold_golden.bin` golden.

Run:
  bazel test //python:coding_oracle_test
"""
import sys
from pathlib import Path

import numpy as np
import frx

frx.config.update("jax_enable_x64", True)
import frx.numpy as jnp  # noqa: E402
from zorch.coding.additive_reed_solomon import (  # noqa: E402
    AdditiveReedSolomon,
    additive_ntt_twiddles,
)
from flock_zorch import ghash  # noqa: E402

ART = Path(__file__).resolve().parents[4] / "artifacts"


def _load_ntt():
    raw = (ART / "ntt_golden.bin").read_bytes()
    assert raw[:8] == b"FLKNTT01", "bad magic"
    log_d = int.from_bytes(raw[8:16], "little")
    n, ntw, off = 1 << log_d, (1 << log_d) - 1, 16
    inp = np.frombuffer(raw, np.uint64, n * 2, off).reshape(n, 2)
    tw = np.frombuffer(raw, np.uint64, ntw * 2, off + n * 16).reshape(ntw, 2)
    out = np.frombuffer(raw, np.uint64, n * 2, off + n * 16 + ntw * 16).reshape(n, 2)
    return log_d, inp, tw, out


def _load_fri():
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
    """uint64 [..., 2] SoA -> binary_field_ghash [...] (via ghash.to_ghash)."""
    return ghash.to_ghash(soa)


def _soa(g):
    """ghash [n] -> uint64 [n, 2] via host bytes (the device ghash->uint64
    bitcast silently returns zeros, zorch#399)."""
    arr = np.asarray(g)
    return np.frombuffer(arr.tobytes(), np.uint64).reshape(arr.shape[0], 2)


def main() -> int:
    print(f"device: {frx.devices()[0]} | backend: {frx.default_backend()}")

    log_d, inp, tw, out = _load_ntt()
    ok = np.array_equal(
        _soa(additive_ntt_twiddles(log_d, jnp.binary_field_ghash)), tw)
    print(f"twiddle table byte-match vs ntt golden (log_d={log_d}): "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        return 1

    # encode = zero-pad message to block_len + LCH NTT; the golden is a full-length
    # transform, so blowup=1 (message_len == block_len == n, no padding).
    code = AdditiveReedSolomon(1 << log_d, 1, jnp.binary_field_ghash)
    got = _soa(frx.jit(code.encode)(_ghash(inp)))
    ok = np.array_equal(got, out)
    print(f"code.encode byte-match vs ntt golden (log_d={log_d}): "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        return 1

    k_code, layer, chal, cw, folded = _load_fri()
    # (2,)-uint64 bitcasts to a 0-d ghash scalar; keep the challenge (1,).
    code = AdditiveReedSolomon(1 << (k_code - 1), 2, jnp.binary_field_ghash)
    cw_g, beta = _ghash(cw), _ghash(chal.reshape(1, 2))
    got = _soa(frx.jit(code.fold)(cw_g, beta))
    ok = np.array_equal(got, folded)
    print(f"code.fold byte-match vs fri golden (k_code={k_code} layer={layer}): "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        return 1

    # fold_values gathers the same pairs code.fold folds — agreement at level 0.
    positions = jnp.arange(min(8, folded.shape[0]))
    fv = code.fold_values(cw_g[2 * positions], cw_g[2 * positions + 1],
                          beta, positions, 0)
    ok = np.array_equal(_soa(fv), folded[np.asarray(positions)])
    print(f"code.fold_values agrees with code.fold: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
