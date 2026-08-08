@echo off
REM Fizgig ROCm Windows installer.
REM Uses detect_gpu.py (GPL-3.0, from comfyui-rocm — see THIRD_PARTY_NOTICES.md)
REM and portable-Python / ROCm-wheel patterns adapted from comfyui-rocm install.bat:
REM   https://github.com/patientx/comfyui-rocm
setlocal enabledelayedexpansion
title Fizgig ROCm Installer
cd /d "%~dp0"

set "Q=>nul 2>&1"
set "PY312="
set "PY312_SOURCE="

echo ============================================================
echo   Fizgig Installer — AMD ROCm (Windows)
echo   Klein 9B and Krea 2 LoRA Studio
echo ============================================================
echo.
echo Uses up-to-date ROCm nightly PyTorch wheels (not Zluda).
echo Requires Python 3.12 ^(the ROCm bitsandbytes wheel is cp312-only^).
echo.

call :resolve_python312
if not defined PY312 (
    echo.
    echo No Python 3.12 found via py launcher or PATH.
    echo Downloading portable Python 3.12.10 ^(same approach as comfyui-rocm^)...
    echo.
    call :download_python312
)
if not defined PY312 (
    echo.
    echo ERROR: Could not locate or install Python 3.12.
    echo Install Python 3.12 from https://www.python.org/downloads/
    echo or ensure `py -3.12` works, then re-run this script.
    pause
    exit /b 1
)

echo Using Python 3.12: !PY312!
echo Source: !PY312_SOURCE!
"!PY312!" --version
echo.

if exist "venv" (
    echo Virtual environment already exists at venv\
    set /p "RECREATE=Delete and recreate? (y/N): "
    if /I "!RECREATE!"=="y" (
        echo Removing existing venv...
        rd /s /q "venv"
    )
)

if not exist "venv" (
    echo Creating virtual environment with Python 3.12...
    "!PY312!" -m venv venv
    if errorlevel 1 (
        echo venv module failed — trying virtualenv...
        "!PY312!" -m pip install virtualenv %Q%
        "!PY312!" -m virtualenv venv
        if errorlevel 1 (
            echo ERROR: Failed to create venv.
            pause
            exit /b 1
        )
    )
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate venv.
    pause
    exit /b 1
)

REM Sanity-check the venv is actually 3.12 (not whatever `python` on PATH defaults to).
python -c "import sys; v=sys.version_info; assert v.major==3 and v.minor==12, f'Expected 3.12, got {v.major}.{v.minor}'; print(f'Venv OK: Python {sys.version.split()[0]}')"
if errorlevel 1 (
    echo ERROR: venv is not Python 3.12 — delete venv\ and re-run.
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip
python -m pip install --upgrade uv

echo.
echo Detecting AMD GPU architecture...
set "arch="
if not exist "detect_gpu.py" (
    echo ERROR: detect_gpu.py not found in %~dp0
    pause
    exit /b 1
)

for /f "delims=" %%A in ('python detect_gpu.py 2^>"%~dp0gpu_detect_debug.log"') do (
    if not "%%A"=="" set "arch=%%A"
)

if "!arch!"=="" (
    echo ERROR: GPU detection failed or unsupported AMD GPU.
    type "%~dp0gpu_detect_debug.log" 2>nul
    pause
    exit /b 1
)

echo Detected GPU architecture: !arch!
echo.

set "USE_LEGACY_URL=0"
for %%G in (gfx942 gfx950) do (
    if /I "!arch!"=="%%G" set "USE_LEGACY_URL=1"
)

if !USE_LEGACY_URL!==1 (
    if /I "!arch!"=="gfx942" (
        echo Installing ROCm PyTorch for MI300/MI325 ^(gfx942^)...
        python -m pip install rocm[devel,libraries] --index-url https://rocm.nightlies.amd.com/v2-staging/gfx942-dcgpu/
        if errorlevel 1 goto :install_failed
        rocm-sdk init
        python -m pip install --index-url https://rocm.nightlies.amd.com/v2-staging/gfx942-dcgpu/ torch torchaudio torchvision
        if errorlevel 1 goto :install_failed
    )
    if /I "!arch!"=="gfx950" (
        echo Installing ROCm PyTorch for MI350/MI355 ^(gfx950^)...
        python -m pip install rocm[devel,libraries] --index-url https://rocm.nightlies.amd.com/v2-staging/gfx950-dcgpu/
        if errorlevel 1 goto :install_failed
        rocm-sdk init
        python -m pip install --index-url https://rocm.nightlies.amd.com/v2-staging/gfx950-dcgpu/ torch torchaudio torchvision
        if errorlevel 1 goto :install_failed
    )
    goto :install_global
)

echo Installing ROCm PyTorch ^(multi-arch nightly^) for !arch!...
python -m pip install "torch[device-!arch!]" "torchvision[device-!arch!]" torchaudio rocm-sdk-devel --index-url https://rocm.nightlies.amd.com/whl-multi-arch/
if errorlevel 1 goto :install_failed

:install_global
echo.
echo Installing Fizgig dependencies...
python -m uv pip install --link-mode copy --index-strategy unsafe-best-match -r requirements-global.txt
if errorlevel 1 goto :install_failed

echo Installing triton-windows ^(torch.compile^)...
python -m pip install triton-windows onnxruntime
if errorlevel 1 goto :install_failed

echo Installing bitsandbytes ^(ROCm wheel^)...
python -m pip install https://github.com/0xDELUXA/bitsandbytes_win_rocm/releases/download/0.50.0.dev0-py3.12-rocm7.15-win_amd64_all/bitsandbytes-0.50.0.dev0-cp312-cp312-win_amd64.whl
if errorlevel 1 goto :install_failed

echo.
echo Verifying ROCm / HIP...
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'GPU available: {torch.cuda.is_available()}'); print(f'HIP: {getattr(torch.version, \"hip\", \"n/a\")}'); print(f'ROCm: {getattr(torch.version, \"rocm\", \"n/a\")}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
if errorlevel 1 goto :install_failed

echo.
echo Writing ROCm launcher config ^(BNB_ROCM_VERSION for bitsandbytes^)...
python write_rocm_env.py
if errorlevel 1 goto :install_failed

echo.
echo Downloading InsightFace models ^(CPU, ~300 MB^)...
python -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; from insightface.app import FaceAnalysis; app=FaceAnalysis(name='buffalo_l', allowed_modules=['detection','genderage','recognition'], providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1); print('Models ready.')"

echo.
echo ============================================================
echo   Installation complete!
echo.
echo   Launch with: run_fizgig_rocm.bat
echo   ^(sets ROCm tuning env vars before starting the GUI^)
echo ============================================================
goto :end

:install_failed
echo.
echo ERROR: Installation failed. See messages above.
pause
exit /b 1

:end
pause
exit /b 0


REM ---------------------------------------------------------------------------
REM Resolve Python 3.12 — never trust bare `python` when 3.14+ is the default.
REM Order: py -3.12  >  python3.12  >  where python3.12  >  existing python312\
REM ---------------------------------------------------------------------------
:resolve_python312
if exist "%~dp0python312\python.exe" (
    call :verify_python312 "%~dp0python312\python.exe" "portable download (python312\)"
    if defined PY312 exit /b 0
)

REM Python launcher — handles multi-version installs (py list).
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -V %Q%
    if not errorlevel 1 (
        for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do (
            call :verify_python312 "%%P" "py launcher (py -3.12)"
            if defined PY312 exit /b 0
        )
    )
)

REM Explicit python3.12 on PATH (OneTrainer install.custom.bat style).
for %%C in (python3.12 python3.12.exe) do (
    where %%C >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%P in ('where %%C 2^>nul') do (
            call :verify_python312 "%%P" "PATH (%%C)"
            if defined PY312 exit /b 0
        )
    )
)

REM Last resort: default `python` only if it is actually 3.12.x.
where python >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        call :verify_python312 "%%P" "PATH (python)"
        if defined PY312 exit /b 0
    )
)
exit /b 0


:verify_python312
set "_CAND=%~1"
set "_SRC=%~2"
"%_CAND%" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)" %Q%
if errorlevel 1 exit /b 0
set "PY312=%_CAND%"
set "PY312_SOURCE=%_SRC%"
exit /b 0


REM ---------------------------------------------------------------------------
REM Download portable Python 3.12.10 (embed + dev libs), same pattern as comfyui-rocm.
REM Stored in python312\ — reused on later installs.
REM ---------------------------------------------------------------------------
:download_python312
set "PYDIR=%~dp0python312"
set "PYVER=3.12.10"

if exist "%PYDIR%\python.exe" (
    call :verify_python312 "%PYDIR%\python.exe" "portable download (python312\)"
    if defined PY312 exit /b 0
)

if not exist "%PYDIR%" mkdir "%PYDIR%"

echo [1/5] Downloading Python %PYVER% embeddable...
curl -L --ssl-no-revoke "https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-embed-amd64.zip" -o "%TEMP%\fizgig_python_embed.zip" --no-progress-meter %Q%
if errorlevel 1 (
    echo ERROR: Failed to download Python embeddable.
    exit /b 1
)

echo [2/5] Downloading Python %PYVER% development files...
curl -L --ssl-no-revoke "https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-amd64.zip" -o "%TEMP%\fizgig_python_full.zip" --no-progress-meter %Q%
if errorlevel 1 (
    echo ERROR: Failed to download Python full zip.
    del "%TEMP%\fizgig_python_embed.zip" %Q%
    exit /b 1
)

echo [3/5] Extracting Python runtime...
tar -xf "%TEMP%\fizgig_python_embed.zip" -C "%PYDIR%" %Q%
if errorlevel 1 (
    echo ERROR: Failed to extract embeddable Python.
    exit /b 1
)
del "%TEMP%\fizgig_python_embed.zip" %Q%

echo [4/5] Copying headers and libraries...
set "PYFULL=%TEMP%\fizgig_pythonfull"
if exist "%PYFULL%" rd /s /q "%PYFULL%" %Q%
mkdir "%PYFULL%"
tar -xf "%TEMP%\fizgig_python_full.zip" -C "%PYFULL%" %Q%
del "%TEMP%\fizgig_python_full.zip" %Q%

if exist "%PYFULL%\include" xcopy "%PYFULL%\include" "%PYDIR%\include\" /E /I /Q %Q%
if exist "%PYFULL%\libs" xcopy "%PYFULL%\libs" "%PYDIR%\libs\" /E /I /Q %Q%
if exist "%PYFULL%\Lib" xcopy "%PYFULL%\Lib" "%PYDIR%\Lib\" /E /I /Q %Q%
rd /s /q "%PYFULL%" %Q%

(
echo python312.zip
echo .
echo ..
echo import site
) > "%PYDIR%\python312._pth"

echo [5/5] Installing pip into portable Python...
curl -L --ssl-no-revoke "https://bootstrap.pypa.io/get-pip.py" -o "%TEMP%\fizgig_get-pip.py" --no-progress-meter %Q%
if errorlevel 1 (
    echo ERROR: Failed to download get-pip.py.
    exit /b 1
)
"%PYDIR%\python.exe" "%TEMP%\fizgig_get-pip.py" --no-warn-script-location %Q%
del "%TEMP%\fizgig_get-pip.py" %Q%
if errorlevel 1 (
    echo ERROR: Failed to install pip into portable Python.
    exit /b 1
)

"%PYDIR%\python.exe" -m pip install --upgrade pip setuptools wheel --no-warn-script-location %Q%

call :verify_python312 "%PYDIR%\python.exe" "portable download (python312\)"
exit /b 0
