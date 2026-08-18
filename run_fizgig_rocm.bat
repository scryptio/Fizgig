@echo off
REM Fizgig launcher for AMD ROCm on Windows.
REM Sets ROCm/HIP tuning env vars, then starts the consoleless GUI chain.
cd /d "%~dp0"

echo [AMD-ROCm] Setting environment variables...
set "MIOPEN_FIND_MODE=2"
set "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE"
REM expandable_segments mirrors upstream train scripts' CUDA alloc policy (ROCm key).
set "PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8"
set "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
set "FIZGIG_GPU_BACKEND=rocm"

REM BNB_ROCM_VERSION / ROCM_PATH / HIP_PATH - written by install_fizgig_rocm.bat
REM to match the installed PyTorch ROCm SDK (e.g. 715 for ROCm 7.15).
if exist "%~dp0rocm_env.bat" (
    call "%~dp0rocm_env.bat"
) else if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0write_rocm_env.py" >nul 2>&1
    if exist "%~dp0rocm_env.bat" call "%~dp0rocm_env.bat"
)
if not defined BNB_ROCM_VERSION set "BNB_ROCM_VERSION=715"
if not defined ROCM_PATH set "ROCM_PATH=%~dp0venv\Lib\site-packages\_rocm_sdk_core"
if not defined HIP_PATH set "HIP_PATH=%ROCM_PATH%"

REM bitsandbytes cuda_specs runs "hipinfo" on Windows — lives under the pip ROCm SDK bin/ and venv\Scripts.
if defined ROCM_PATH if exist "%ROCM_PATH%\bin" set "PATH=%ROCM_PATH%\bin;%PATH%"
if exist "%~dp0venv\Lib\site-packages\_rocm_sdk_devel\bin" set "PATH=%~dp0venv\Lib\site-packages\_rocm_sdk_devel\bin;%PATH%"
if exist "%~dp0venv\Scripts" set "PATH=%~dp0venv\Scripts;%PATH%"

echo [AMD-ROCm] BNB_ROCM_VERSION=%BNB_ROCM_VERSION%  ROCM_PATH=%ROCM_PATH%

start "" /b wscript //nologo //b "%~dp0run_silent.vbs"
