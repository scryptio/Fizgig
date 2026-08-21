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

REM BNB_ROCM_VERSION / ROCM_PATH / HIP_PATH / gfx12 batched-GEMM wa - written by
REM install_fizgig_rocm.bat / write_rocm_env.py (not on every launch; re-run write_rocm_env.py
REM after a pull if rocm_env.bat is stale).
if exist "%~dp0rocm_env.bat" (
    call "%~dp0rocm_env.bat"
) else if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0write_rocm_env.py" >nul 2>&1
    if exist "%~dp0rocm_env.bat" call "%~dp0rocm_env.bat"
)
REM Pinned installs set BNB_ROCM_VERSION in rocm_env.bat. Floating --experimental leaves it
REM unset so bitsandbytes picks its highest matching DLL (override: set BNB_ROCM_VERSION).
if not defined ROCM_PATH set "ROCM_PATH=%~dp0venv\Lib\site-packages\_rocm_sdk_core"
if not defined HIP_PATH set "HIP_PATH=%ROCM_PATH%"

REM bitsandbytes cuda_specs runs "hipinfo" on Windows - lives under the pip ROCm SDK bin/ and venv\Scripts.
if defined ROCM_PATH if exist "%ROCM_PATH%\bin" set "PATH=%ROCM_PATH%\bin;%PATH%"
if exist "%~dp0venv\Lib\site-packages\_rocm_sdk_devel\bin" set "PATH=%~dp0venv\Lib\site-packages\_rocm_sdk_devel\bin;%PATH%"
if exist "%~dp0venv\Scripts" set "PATH=%~dp0venv\Scripts;%PATH%"

REM Opt out of gfx12 ROCBLAS_USE_HIPBLASLT_BATCHED=0 from rocm_env.bat:
REM   set FIZGIG_NO_ROCBLAS_BATCHED_WA=1
if defined FIZGIG_NO_ROCBLAS_BATCHED_WA set "ROCBLAS_USE_HIPBLASLT_BATCHED="

if defined BNB_ROCM_VERSION (
    echo [AMD-ROCm] BNB_ROCM_VERSION=%BNB_ROCM_VERSION%  ROCM_PATH=%ROCM_PATH%
) else (
    echo [AMD-ROCm] BNB_ROCM_VERSION unset ^(bitsandbytes selects DLL^)  ROCM_PATH=%ROCM_PATH%
)
if defined ROCBLAS_USE_HIPBLASLT_BATCHED (
    echo [AMD-ROCm] ROCBLAS_USE_HIPBLASLT_BATCHED=%ROCBLAS_USE_HIPBLASLT_BATCHED% ^(gfx12 batched GEMM wa^)
)

start "" /b wscript //nologo //b "%~dp0run_silent.vbs"
