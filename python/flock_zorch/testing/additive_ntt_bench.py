"""Additive-NTT benchmark: the compiler's fused `lax.ntt` (`ntt_pass_fusion`),
plus a byte-match gate against binius-gpu's `additive_ntt_kernel` at GF(2^32).

Context (flock-zorch #23): the premise for a hand-written FFI kernel (#22) was
that XLA de-fuses the additive-NTT butterfly network. It does not — `lax.ntt` on
binary-field dtypes lowers to a few fused `ntt_pass_fusion` custom kernels. This
bench:

  1. proves that fused kernel is byte-identical to binius-gpu's hand-written
     `additive_ntt_kernel<uint32, FanPaarTowerField<5>>` (GF(2^32), the one width
     binius supports) — verified against binius's own pinned MD5 oracle; and
  2. measures its throughput, which plateaus at only ~3-5% of the XOR-add
     bandwidth ceiling — i.e. the transform is field-multiply-bound, not
     bandwidth-bound. See optim/additive_ntt/README.md for the head-to-head vs
     binius's kernel timing and the compute-bound analysis.

Kernel-only device time is measured async-pipelined: dispatch `iters` identical
ops without blocking (independent dispatches are not fused, so the XLA stream
runs them back-to-back and host dispatch overlaps execution) and block once —
matching binius's CUDA-event-bracketed kernel loop.

Run:  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
        .venv/bin/python -m flock_zorch.testing.additive_ntt_bench
"""
import hashlib
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

# binius-gpu's pinned rate-0 output MD5s (src/ulvt/ntt/tests/test_ntt.cu,
# additive_ntt_hashes[0][log_h]; "pulled from the python model and risc0").
# Reproducing binius's mt19937 input in lax.ntt(binary_field_t5) must match these.
BINIUS_ADDITIVE_NTT_MD5_R0 = {
    1: "6c674a56275dfd6baf965163d6d4757a",
    2: "373b753b3e053d128cb53ac23f403a1c",
    3: "0933fa26689378684a4f6a4654deed44",
    4: "3f8d244dc683e58534c8a1bef2284127",
    5: "2f72470ce905c9261380bac9232db7ae",
    6: "a22e4b3ae73b2a7c4443288e7f8fdfca",
    7: "81179f7e33b4522b20bacba9c07db9cd",
    8: "fb4c3004906ef7d59d5c5a5a0485e290",
    9: "d04bcce5c7d1a85995a9e9a654b58323",
    10: "191e2bc2ee655300c27f7c244952c0b7",
}


class MT19937:
    """std::mt19937 with libstdc++ single-uint32-seed init — the exact input
    stream binius uses (`std::mt19937 gen(0xdeadbeef + log_h + log_rate)`)."""

    def __init__(self, seed):
        self.mt = [seed & 0xFFFFFFFF] + [0] * 623
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.idx = 624

    def _gen(self):
        for i in range(624):
            y = (self.mt[i] & 0x80000000) + (self.mt[(i + 1) % 624] & 0x7FFFFFFF)
            self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
        self.idx = 0

    def next(self):
        if self.idx >= 624:
            self._gen()
        y = self.mt[self.idx]
        self.idx += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF


def _mt_u32(seed, n):
    g = MT19937(seed)
    return np.fromiter((g.next() for _ in range(n)), dtype=np.uint32, count=n)


def _rand_field(log_h, dtype, fast=False):
    """Deterministic random `2^log_h`-element binary-field array. `fast=True` uses a
    numpy RNG for the timing sweep — the NTT is data-independent, so exact inputs
    don't matter there. `fast=False` reproduces binius's `std::mt19937` stream, which
    the byte-match gate needs (and which is ~tens of seconds in pure Python at
    128-bit/2^23, so it is used only for the small gate)."""
    n = 1 << log_h
    words_per = np.dtype(dtype).itemsize // 4
    m = n * words_per
    words = (np.random.default_rng(0xDEADBEEF + log_h).integers(0, 2**32, size=m, dtype=np.uint32)
             if fast else _mt_u32(0xDEADBEEF + log_h, m))
    return lax.bitcast_convert_type(jnp.asarray(words).reshape(n, words_per), dtype).reshape(-1)


def _time_ms(fn, args, iters):
    """Async-pipelined device time (ms/op): dispatch `iters` identical ops, block
    once. Independent dispatches are not fused, so each is a real device op; the
    XLA stream runs them back-to-back and host dispatch overlaps execution."""
    f = jax.jit(fn)
    f(*args).block_until_ready()  # compile + warm
    t0 = time.perf_counter()
    r = None
    for _ in range(iters):
        r = f(*args)
    r.block_until_ready()
    return (time.perf_counter() - t0) / iters * 1e3


def byte_match_gate():
    """lax.ntt(binary_field_t5) reproduces binius's additive_ntt_kernel output
    (verified against binius's pinned MD5 oracle) — proving the GF(2^32) tower
    encoding + LCH domain + normalization are bit-identical across the two."""
    f = jax.jit(lambda x: lax.ntt(x, ntt_type="NTT", ntt_length=x.shape[-1]))
    ok = 0
    for log_h, expect in BINIUS_ADDITIVE_NTT_MD5_R0.items():
        z = _rand_field(log_h, jnp.binary_field_t5)
        out = np.asarray(lax.bitcast_convert_type(f(z), jnp.uint32)).astype("<u4")
        ok += hashlib.md5(out.tobytes()).hexdigest() == expect
    n = len(BINIUS_ADDITIVE_NTT_MD5_R0)
    print(f"byte-match vs binius (GF(2^32), rate 0): {ok}/{n}" + ("  OK" if ok == n else "  MISMATCH"))
    return ok == n


def sweep(dtype_name, logs, iters=40):
    dtype = getattr(jnp, dtype_name)
    print(f"\n=== lax.ntt {dtype_name} ({np.dtype(dtype).itemsize * 8}-bit), rate 0 ===")
    print(f"{'log_d':>5} {'ntt_ms':>9} {'Gelem/s':>9}")
    for log_h in logs:
        n = 1 << log_h
        try:
            z = _rand_field(log_h, dtype, fast=True)
            ntt_ms = _time_ms(lambda x: lax.ntt(x, ntt_type="NTT", ntt_length=n), (z,), iters)
            print(f"{log_h:>5} {ntt_ms:>9.4f} {n / (ntt_ms / 1e3) / 1e9:>9.3f}")
            del z
        except Exception as e:  # noqa: BLE001
            print(f"{log_h:>5}   FAIL: {type(e).__name__}: {str(e)[:70]}")


def main():
    print("device:", jax.devices()[0].device_kind)
    byte_match_gate()
    logs = range(10, 24)
    sweep("binary_field_t5", logs)     # GF(2^32) — the binius head-to-head width
    sweep("binary_field_ghash", logs)  # GF(2^128) — flock's actual field
    sweep("binary_field_t7", logs)     # GF(2^128) tower — cross-check


if __name__ == "__main__":
    main()
