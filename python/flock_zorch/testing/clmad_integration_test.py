"""clmad integration gate + benchmark.

Validates that the FFI clmad path is byte-identical to flock both for the bare
multiply (`field_clmad.mul`) and inside the additive NTT (`forward_transform_scalar`
with `mul=field_clmad.mul`), then measures the NTT speedup from clmad. Needs the
built FFI handler (optim/clmad/build_ffi.sh) + an sm_120 GPU.
"""
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from flock_zorch import field, field_clmad, ntt as ntt_mod  # noqa: E402

ART = Path(__file__).resolve().parents[3] / "artifacts"


def _load_field_golden():
    raw = (ART / "field_mul_golden.bin").read_bytes()
    n = int.from_bytes(raw[8:16], "little")
    off, blk = 16, n * 16
    g = lambda o: np.frombuffer(raw, np.uint64, n * 2, o).reshape(n, 2)  # noqa: E731
    return n, g(off), g(off + blk), g(off + 2 * blk)


def _load_ntt_golden():
    raw = (ART / "ntt_golden.bin").read_bytes()
    log_d = int.from_bytes(raw[8:16], "little")
    n, ntw, off = 1 << log_d, (1 << log_d) - 1, 16
    inp = np.frombuffer(raw, np.uint64, n * 2, off).reshape(n, 2)
    tw = np.frombuffer(raw, np.uint64, ntw * 2, off + n * 16).reshape(ntw, 2)
    out = np.frombuffer(raw, np.uint64, n * 2, off + n * 16 + ntw * 16).reshape(n, 2)
    return log_d, inp, tw, out


def test_field_clmad_oracle():
    n, a, b, golden = _load_field_golden()
    got = np.asarray(jax.jit(field_clmad.mul)(jnp.asarray(a), jnp.asarray(b)))
    assert np.array_equal(got, golden), "field_clmad.mul != flock golden"


def test_ntt_clmad_oracle():
    log_d, inp, tw, golden = _load_ntt_golden()
    fn = jax.jit(lambda d, t: ntt_mod.forward_transform_scalar(d, t, log_d, mul=field_clmad.mul))
    got = np.asarray(fn(jnp.asarray(inp), jnp.asarray(tw)))
    assert np.array_equal(got, golden), "NTT(clmad) != flock golden"


def main():
    if not field_clmad.available():
        print("clmad FFI not built — run optim/clmad/build_ffi.sh")
        return
    test_field_clmad_oracle()
    print("field_clmad == flock golden: PASS")
    test_ntt_clmad_oracle()
    print("NTT(clmad)  == flock golden: PASS")

    print("\nNTT forward_transform_scalar: software-mul vs clmad-mul")
    for log in (16, 18, 20):
        n = 1 << log
        d = jnp.asarray(np.random.default_rng(3).integers(0, 2**64, size=(n, 2), dtype=np.uint64))
        tw = jnp.asarray(np.random.default_rng(4).integers(0, 2**64, size=(n - 1, 2), dtype=np.uint64))
        row = f"  log_d={log:<2}"
        for name, mulf in (("software", field.mul), ("clmad", field_clmad.mul)):
            fn = jax.jit(lambda dd, tt, m=mulf: ntt_mod.forward_transform_scalar(dd, tt, log, mul=m))
            r = fn(d, tw)
            r.block_until_ready()
            it = 30
            t0 = time.perf_counter()
            for _ in range(it):
                r = fn(d, tw)
            r.block_until_ready()
            row += f"  {name}={(time.perf_counter()-t0)/it*1e3:7.2f}ms"
        print(row)


if __name__ == "__main__":
    main()
