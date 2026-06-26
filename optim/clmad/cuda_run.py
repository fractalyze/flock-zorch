"""Minimal CUDA driver-API launcher via ctypes -> libcuda.so.1.

Loads a ptxas-built cubin (SASS, no JIT), runs a kernel, copies back. Used to
validate + benchmark the PTX `clmad` GHASH multiply on sm_120 without nvcc/cudart.
"""
import ctypes
import time

cu = ctypes.CDLL("libcuda.so.1")
_void = ctypes.c_void_p
_u32 = ctypes.c_uint32
_sz = ctypes.c_size_t
P = ctypes.POINTER

cu.cuGetErrorString.argtypes = [ctypes.c_int, P(ctypes.c_char_p)]
cu.cuInit.argtypes = [_u32]
cu.cuDeviceGet.argtypes = [P(ctypes.c_int), ctypes.c_int]
cu.cuCtxCreate_v2.argtypes = [P(_void), _u32, ctypes.c_int]
cu.cuModuleLoadData.argtypes = [P(_void), _void]
cu.cuModuleGetFunction.argtypes = [P(_void), _void, ctypes.c_char_p]
cu.cuMemAlloc_v2.argtypes = [P(_void), _sz]
cu.cuMemcpyHtoD_v2.argtypes = [_void, _void, _sz]
cu.cuMemcpyDtoH_v2.argtypes = [_void, _void, _sz]
cu.cuLaunchKernel.argtypes = [_void, _u32, _u32, _u32, _u32, _u32, _u32,
                              _u32, _void, P(_void), P(_void)]
cu.cuCtxSynchronize.argtypes = []
cu.cuMemFree_v2.argtypes = [_void]


def _ck(err, what):
    if err != 0:
        s = ctypes.c_char_p()
        cu.cuGetErrorString(err, ctypes.byref(s))
        raise RuntimeError(f"CUDA {err} ({(s.value or b'?').decode()}) @ {what}")


def init(dev=0):
    _ck(cu.cuInit(0), "cuInit")
    d = ctypes.c_int()
    _ck(cu.cuDeviceGet(ctypes.byref(d), dev), "cuDeviceGet")
    ctx = _void()
    _ck(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, d), "cuCtxCreate")
    return ctx


def load(cubin_path):
    data = open(cubin_path, "rb").read()
    mod = _void()
    _ck(cu.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(data)), "cuModuleLoadData")
    return mod


def func(mod, name):
    f = _void()
    _ck(cu.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()), "cuModuleGetFunction")
    return f


def alloc(nbytes):
    p = _void()
    _ck(cu.cuMemAlloc_v2(ctypes.byref(p), nbytes), "cuMemAlloc")
    return p


def htod(dptr, hostbytes):
    buf = (ctypes.c_char * len(hostbytes)).from_buffer_copy(hostbytes)
    _ck(cu.cuMemcpyHtoD_v2(dptr, buf, len(hostbytes)), "HtoD")


def dtoh(dptr, nbytes):
    buf = (ctypes.c_char * nbytes)()
    _ck(cu.cuMemcpyDtoH_v2(buf, dptr, nbytes), "DtoH")
    return bytes(buf)


def launch(f, grid, block, args):
    """args: list of ctypes scalar values (already the right type)."""
    arr = (_void * len(args))()
    for i, a in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a), _void)
    _ck(cu.cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, None, arr, None), "launch")


def sync():
    _ck(cu.cuCtxSynchronize(), "sync")


if __name__ == "__main__":
    import sys
    cubin = sys.argv[1] if len(sys.argv) > 1 else "trivial.cubin"
    init()
    mod = load(cubin)
    f = func(mod, "writeidx")
    n = 1 << 20
    out = alloc(n * 8)
    block = 256
    grid = (n + block - 1) // block
    launch(f, grid, block, [out, _u32(n)])
    sync()
    raw = dtoh(out, n * 8)
    import struct
    vals = struct.unpack("<8Q", raw[:64])
    ok = vals == tuple(range(8)) and struct.unpack("<Q", raw[-8:])[0] == n - 1
    print("trivial launch:", "PASS" if ok else f"FAIL {vals}")
