#!/usr/bin/env bash
# Build the clmad GHASH-multiply XLA FFI handler (libghash_clmad.so) + the cubin.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/home/jooman/fractalyze/zorch/.venv/bin/python
INC="$("$VENV" -c 'import jax.ffi; print(jax.ffi.include_dir())')"
PTXAS="$HOME/.local/cuda13/bin/ptxas"
# driver-API header (stable across versions) + link stub from the system CUDA 12.9
CUDA_INC=/usr/local/cuda-12.9/include
CUDA_STUB=/usr/local/cuda-12.9/lib64/stubs

echo "== sanity =="
grep -q "XLA_FFI_DEFINE_HANDLER_SYMBOL" "$INC/xla/ffi/api/api.h" "$INC/xla/ffi/api/ffi.h" \
  && echo "  macro found" || echo "  WARN: XLA_FFI_DEFINE_HANDLER_SYMBOL not found"
[ -f "$CUDA_INC/cuda.h" ] && echo "  cuda.h ok" || { echo "  NO cuda.h at $CUDA_INC"; exit 1; }

echo "== assemble clmad cubin (ptxas 13.3, sm_120) =="
"$PTXAS" -arch=sm_120 -O3 "$HERE/ghash_mul.ptx" -o "$HERE/ghash_mul.cubin"

echo "== compile FFI handler -> libghash_clmad.so =="
g++ -O3 -fPIC -shared -std=c++17 \
  -I"$INC" -I"$CUDA_INC" \
  "$HERE/ghash_mul_ffi.cpp" -o "$HERE/libghash_clmad.so" \
  -L"$CUDA_STUB" -lcuda

echo "== exported symbol =="
nm -D "$HERE/libghash_clmad.so" | grep -i GhashMul || echo "  (no GhashMul symbol!)"
echo "OK: $HERE/libghash_clmad.so"
