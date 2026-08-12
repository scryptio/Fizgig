#!/usr/bin/env bash
# Fizgig launcher for AMD ROCm on Linux — mirrors run_fizgig_rocm.bat env tuning.
# HIGHLY EXPERIMENTAL: Linux AMD training is best-effort only.
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source venv/bin/activate

export MIOPEN_FIND_MODE=2
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export FIZGIG_GPU_BACKEND=rocm

# BNB_ROCM_VERSION / ROCM_PATH / HIP_PATH — written by install_fizgig_rocm_linux.sh / write_rocm_env.py
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

if [[ -z "${BNB_ROCM_VERSION:-}" ]]; then
    export BNB_ROCM_VERSION=714  # libbitsandbytes_rocm714.so — Linux stable torch stack
fi
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
            /opt/rocm/lib /opt/rocm/lib64; do
    [[ -d "$_lib" ]] && LD_LIBRARY_PATH="${_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH

echo "[AMD-ROCm] BNB_ROCM_VERSION=${BNB_ROCM_VERSION:-}  ROCM_PATH=${ROCM_PATH:-}"

python lora_trainer_gui.py &
disown
