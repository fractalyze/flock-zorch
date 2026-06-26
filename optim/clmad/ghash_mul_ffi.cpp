// XLA FFI handler that runs flock's GF(2^128) GHASH multiply via the PTX `clmad`
// cubin on XLA's GPU stream. Lets the jax-exported prover call clmad for field.mul
// without rebuilding the zkx plugin: jax `field.mul` -> ffi_call -> this handler ->
// clmad kernel (optim/clmad/ghash_mul.ptx, assembled by ptxas 13.3).
//
// Build: see build_ffi.sh. Register from jax with jax.ffi.register_ffi_target.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Hoist comma-bearing template types into aliases so the FFI macro (which splits
// its args on commas) doesn't choke on `Buffer<DataType::U64, 2>`.
using BufU64 = ffi::Buffer<ffi::DataType::U64, 2>;
using ResU64 = ffi::Result<ffi::Buffer<ffi::DataType::U64, 2>>;

static CUfunction g_func = nullptr;
static std::once_flag g_once;

static void load_kernel() {
  cuInit(0);
  const char* path = std::getenv("FLOCK_CLMAD_CUBIN");
  std::string cubin = path ? path
      : "/home/jooman/fractalyze/flock-zorch/optim/clmad/ghash_mul.cubin";
  FILE* f = std::fopen(cubin.c_str(), "rb");
  if (!f) { std::fprintf(stderr, "clmad-ffi: cannot open %s\n", cubin.c_str()); std::abort(); }
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<char> buf(sz);
  if (std::fread(buf.data(), 1, sz, f) != (size_t)sz) { std::abort(); }
  std::fclose(f);
  CUmodule mod;
  CUresult r = cuModuleLoadData(&mod, buf.data());
  if (r != CUDA_SUCCESS) {
    const char* e; cuGetErrorString(r, &e);
    std::fprintf(stderr, "clmad-ffi: cuModuleLoadData: %s\n", e); std::abort();
  }
  r = cuModuleGetFunction(&g_func, mod, "ghash_mul");
  if (r != CUDA_SUCCESS) { std::fprintf(stderr, "clmad-ffi: no ghash_mul\n"); std::abort(); }
}

static ffi::Error GhashMulImpl(CUstream stream, BufU64 a, BufU64 b, ResU64 out) {
  std::call_once(g_once, load_kernel);
  auto dims = a.dimensions();
  unsigned n = static_cast<unsigned>(dims[0]);  // [N, 2]
  CUdeviceptr da = reinterpret_cast<CUdeviceptr>(a.typed_data());
  CUdeviceptr db = reinterpret_cast<CUdeviceptr>(b.typed_data());
  CUdeviceptr dout = reinterpret_cast<CUdeviceptr>(out->typed_data());
  unsigned block = 256;
  unsigned grid = (n + block - 1) / block;
  void* args[] = {&da, &db, &dout, &n};
  CUresult r = cuLaunchKernel(g_func, grid, 1, 1, block, 1, 1, 0, stream, args, nullptr);
  if (r != CUDA_SUCCESS) {
    const char* e; cuGetErrorString(r, &e);
    return ffi::Error(ffi::ErrorCode::kInternal, std::string("clmad launch: ") + e);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GhashMul, GhashMulImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<CUstream>>()
        .Arg<BufU64>()
        .Arg<BufU64>()
        .Ret<BufU64>());
