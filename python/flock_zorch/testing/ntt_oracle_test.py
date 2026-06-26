"""Byte-match gate for the additive NTT over F128 (port target #2).

Reads (input, twiddles, output) produced by flock-core's reference
`forward_transform_scalar` (`examples/dump_ntt.rs`) and asserts the jax port in
`flock_zorch.ntt.forward_transform_scalar` reproduces the output byte-for-byte.

Run directly (backend chosen by JAX_PLATFORMS):
    JAX_PLATFORMS=cuda PYTHONPATH=python <venv>/bin/python \
        python/flock_zorch/testing/ntt_oracle_test.py
"""
import os
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flock_zorch import ntt as ntt_mod  # noqa: E402

_MAGIC = b"FLKNTT01"


def _artifacts_dir() -> Path:
    env = os.environ.get("FLOCK_ZORCH_ARTIFACTS")
    return Path(env) if env else Path(__file__).resolve().parents[3] / "artifacts"


def _load(path: Path):
    raw = path.read_bytes()
    assert raw[:8] == _MAGIC, f"bad magic {raw[:8]!r} in {path}"
    log_d = int.from_bytes(raw[8:16], "little")
    n = 1 << log_d
    ntw = n - 1
    off = 16
    inp = np.frombuffer(raw, np.uint64, count=n * 2, offset=off).reshape(n, 2)
    off += n * 16
    tw = np.frombuffer(raw, np.uint64, count=ntw * 2, offset=off).reshape(ntw, 2)
    off += ntw * 16
    out = np.frombuffer(raw, np.uint64, count=n * 2, offset=off).reshape(n, 2)
    return log_d, inp, tw, out


def run(path: Path | None = None):
    path = path or (_artifacts_dir() / "ntt_golden.bin")
    log_d, inp, tw_golden, golden = _load(path)

    # 1. flock-zorch computes its OWN twiddles (host) — must match flock's exactly.
    tw = ntt_mod.compute_twiddles(log_d)
    if not np.array_equal(tw, tw_golden):
        diff = np.flatnonzero(np.any(tw != tw_golden, axis=1))
        i = int(diff[0])
        raise AssertionError(
            f"twiddle mismatch {len(diff)}/{len(tw_golden)}; first @ {i}: "
            f"computed={tw[i].tolist()} golden={tw_golden[i].tolist()}"
        )

    # 2. the transform (using our computed twiddles) must reproduce flock's output.
    fn = jax.jit(lambda d, t: ntt_mod.forward_transform_scalar(d, t, log_d))
    got = np.asarray(fn(jnp.asarray(inp), jnp.asarray(tw)))
    if not np.array_equal(got, golden):
        diff = np.flatnonzero(np.any(got != golden, axis=1))
        i = int(diff[0])
        raise AssertionError(
            f"NTT mismatch {len(diff)}/{len(golden)} rows; first @ {i}: "
            f"got={got[i].tolist()} golden={golden[i].tolist()}"
        )
    return log_d, len(golden)


def test_ntt_oracle():
    run()


if __name__ == "__main__":
    ld, n = run()
    print(f"additive-NTT byte-match: PASS on {jax.default_backend()} (log_d={ld}, {n} elems)")
