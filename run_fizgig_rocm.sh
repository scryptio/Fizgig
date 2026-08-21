#!/usr/bin/env bash
# Fizgig launcher for AMD ROCm on Linux - mirrors run_fizgig_rocm.bat env tuning.
# HIGHLY EXPERIMENTAL: Linux AMD training is best-effort only.
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source venv/bin/activate

export MIOPEN_FIND_MODE=2
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
# Allocator: upstream GUI honours FIZGIG_NO_EXPANDABLE=1 (A/B opt-out, lora_trainer_gui.py).
export FIZGIG_NO_EXPANDABLE=1
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export FIZGIG_GPU_BACKEND=rocm
# Cache fast-exit: src/fizgig/rocm/cache_exit.py (Linux ROCm only). Opt out: FIZGIG_ROCM_NO_FAST_EXIT=1

# BNB_ROCM_VERSION / ROCM_PATH / HIP_PATH / gfx12 batched-GEMM wa - write_rocm_env.py
# (install / first launch only; re-run write_rocm_env.py after a pull if rocm_env.sh is stale).
if [[ -f rocm_env.sh ]]; then
    # shellcheck source=rocm_env.sh
    source rocm_env.sh
elif [[ -x venv/bin/python ]]; then
    venv/bin/python write_rocm_env.py >/dev/null 2>&1 || true
    if [[ -f rocm_env.sh ]]; then
        # shellcheck source=rocm_env.sh
        source rocm_env.sh
    fi
fi

# Pinned installs set BNB_ROCM_VERSION in rocm_env.sh; otherwise leave unset for bitsandbytes.
if [[ -z "${ROCM_PATH:-}" ]]; then
    for _p in venv/lib/python*/site-packages/_rocm_sdk_core; do
        if [[ -d "$_p" ]]; then
            export ROCM_PATH="$(cd "$_p" && pwd)"
            break
        fi
    done
fi
if [[ -z "${HIP_PATH:-}" && -n "${ROCM_PATH:-}" ]]; then
    export HIP_PATH="$ROCM_PATH"
fi

for _d in /opt/rocm/core-*/bin /opt/rocm/bin; do
    [[ -d "$_d" ]] && PATH="$_d:$PATH"
done
export PATH

for _lib in venv/lib/python*/site-packages/_rocm_sdk_core/lib \
            venv/lib/python*/site-packages/_rocm_sdk/lib \
            venv/lib/python*/site-packages/_rocm_sdk_libraries/lib \
            /opt/rocm/lib /opt/rocm/lib64; do
    [[ -d "$_lib" ]] && LD_LIBRARY_PATH="${_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH

# Opt out of gfx12 ROCBLAS_USE_HIPBLASLT_BATCHED=0 from rocm_env.sh:
#   FIZGIG_NO_ROCBLAS_BATCHED_WA=1 ./run_fizgig_rocm.sh
if [[ -n "${FIZGIG_NO_ROCBLAS_BATCHED_WA:-}" ]]; then
    unset ROCBLAS_USE_HIPBLASLT_BATCHED
fi

if [[ -n "${BNB_ROCM_VERSION:-}" ]]; then
    echo "[AMD-ROCm] BNB_ROCM_VERSION=${BNB_ROCM_VERSION}  ROCM_PATH=${ROCM_PATH:-}"
else
    echo "[AMD-ROCm] BNB_ROCM_VERSION unset (bitsandbytes selects lib)  ROCM_PATH=${ROCM_PATH:-}"
fi
if [[ -n "${ROCBLAS_USE_HIPBLASLT_BATCHED:-}" ]]; then
    echo "[AMD-ROCm] ROCBLAS_USE_HIPBLASLT_BATCHED=${ROCBLAS_USE_HIPBLASLT_BATCHED} (gfx12 batched GEMM wa)"
fi

python lora_trainer_gui.py &
disown
