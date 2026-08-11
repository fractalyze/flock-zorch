#!/usr/bin/env bash
# ncu the m32 zerocheck multilinear round reduce, with its in-workload
# neighbours in the same capture.
#
# Question this answers: #215 fixed a MEASURED 2.00x traffic amplification in
# this reduce (compiler-side, xla#445). The bucket then fell only 21.8%
# (6.569 -> 5.136 ms). Either the fix is partial, or the amplification is gone
# and the reduce is now at roofline — in which case the ML ladder holds no
# further lever and the next pick is `open`. Those imply different work, and
# the traffic term must NOT be derived from shapes to settle it: #213 tried
# exactly that twice on this exact kernel (56%, then 47%) against an
# ncu-measured 95%. See docs/measurement.md.
#
# Needs root: the driver has RmProfilingAdminOnly=1, so ncu's counters are
# admin-only (ERR_NVGPUCTRPERM) while nsys is not. Run:
#
#   sudo bash scripts/ncu_zerocheck_ml.sh
#
# HOME and every path below are pinned absolute on purpose — sudo's env reset
# otherwise sends CUDA_ROOT and the venv to root's home, and you silently
# measure the software-GHASH path (docs/measurement.md).
set -euo pipefail

REPO=/home/ryan/Workspace/flock-zorch3
OUT="${1:-$REPO/artifacts/ncu_zcml_m32}"

export HOME=/home/ryan
export CUDA_ROOT="$HOME/.local/cuda13-merged"
export PATH="$CUDA_ROOT/bin:/usr/local/bin:/usr/bin:/bin"
export FRX_PLATFORMS=cuda,cpu
export FRX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export FLOCK_ZORCH_ARTIFACTS="$REPO/artifacts"
export PYTHONPATH="$REPO/python:/data/ryan/bazel/0e1988e69a78eafae3d0e32247804dbe/external/zorch+"
unset JAX_PLATFORMS JAX_ENABLE_X64 XLA_PYTHON_CLIENT_ALLOCATOR JAX_COMPILATION_CACHE_DIR

cd "$REPO"
echo "ptxas: $("$CUDA_ROOT/bin/ptxas" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')  (must be 13.3)"

# --profile-from-start no honours the harness's cuProfilerStart/Stop, so warm-up
# and autotune are excluded by construction — the same gate the nsys captures use.
# The kernel filter keeps the fold and the bit-select reduce in the SAME capture
# as in-workload references: siblings are what turn "is this bandwidth good?"
# into a measurement instead of a judgement call.
/opt/nvidia/nsight-compute/2026.2.0/ncu \
  --target-processes all \
  --profile-from-start no \
  --kernel-name 'regex:input_reduce_fusion|loop_add_fusion|bit_select_xor_reduce' \
  --launch-count 16 \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --export "$OUT" --force-overwrite \
  "$REPO/.venv/bin/python" "$REPO/python/flock_zorch/testing/nsys_capture.py" \
    --window zerocheck-ml --golden blake3_ligerito_golden_m32.bin

chown ryan:ryan "$OUT.ncu-rep" 2>/dev/null || true
echo
echo "wrote $OUT.ncu-rep — readable without root via:"
echo "  /opt/nvidia/nsight-compute/2026.2.0/ncu --import $OUT.ncu-rep --page details"
