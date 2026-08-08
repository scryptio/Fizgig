#!/usr/bin/env python3
"""
Fizgig Installer
================
Sets up Fizgig with its own Python virtual environment.
Installs PyTorch for NVIDIA CUDA (default) or AMD ROCm — required for Klein 9B and
Krea 2 training, preview rendering, profiling, extraction, and Repair Studio.

Features:
- Creates isolated venv for Fizgig dependencies
- Installs CUDA 12.8 PyTorch on NVIDIA GPUs (RTX 30xx / 40xx / 50xx Blackwell)
- Installs ROCm PyTorch on Linux AMD GPUs when /dev/kfd is present (or --platform rocm)
- Windows AMD: use install_fizgig_rocm.bat instead (ROCm nightly wheels + GPU detection)
- Installs InsightFace face detection (runs on CPU for GPU independence)
- Downloads face detection models automatically
- Installs Florence-2 AI captioning (transformers library; runs on GPU)
- Creates launcher scripts for Windows/Linux/Mac
- Verifies the GPU backend is visible to PyTorch after install

Note: Florence-2 models are auto-downloaded from Hugging Face on first use (~500MB-1.5GB)
"""

import argparse
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# Minimum Python version required
MIN_PYTHON_VERSION = (3, 10)

# Directory paths
SCRIPT_DIR = Path(__file__).parent.absolute()
VENV_DIR = SCRIPT_DIR / "venv"
REQUIREMENTS_FILE = SCRIPT_DIR / "requirements.txt"
REQUIREMENTS_GLOBAL = SCRIPT_DIR / "requirements-global.txt"
REQUIREMENTS_ROCM_LINUX = SCRIPT_DIR / "requirements-rocm-linux.txt"


def detect_gpu_platform() -> str:
    """Return 'cuda', 'rocm', or 'cpu' based on host hardware."""
    if platform.system() == "Linux":
        if os.path.exists("/dev/nvidia0") or shutil.which("nvidia-smi"):
            return "cuda"
        if os.path.exists("/dev/kfd"):
            return "rocm"
    elif platform.system() == "Windows":
        if shutil.which("nvidia-smi"):
            return "cuda"
        # Windows AMD ROCm uses install_fizgig_rocm.bat (GPU-specific nightly wheels).
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and "AMD" in result.stdout and "Radeon" in result.stdout:
                return "rocm"
        except Exception:
            pass
    return "cpu"


def resolve_platform(requested: str) -> str:
    if requested != "detect":
        return requested
    detected = detect_gpu_platform()
    return detected if detected != "cpu" else "cuda"


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step_num, text):
    """Print a formatted step"""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 40)


def check_python_version():
    """Check if Python version meets minimum requirements"""
    version = sys.version_info[:2]
    if version < MIN_PYTHON_VERSION:
        print(f"Error: Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ required.")
        print(f"Current version: Python {version[0]}.{version[1]}")
        return False
    print(f"Python version: {version[0]}.{version[1]} (OK)")
    return True


def create_venv():
    """Create virtual environment"""
    if VENV_DIR.exists():
        print(f"Virtual environment already exists at: {VENV_DIR}")
        response = input("Delete and recreate? (y/N): ").strip().lower()
        if response == 'y':
            print("Removing existing venv...")
            shutil.rmtree(VENV_DIR)
        else:
            print("Using existing venv.")
            return True

    print(f"Creating virtual environment at: {VENV_DIR}")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print("Virtual environment created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating venv: {e}")
        return False


def get_pip_path():
    """Get the path to pip in the venv"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def get_python_path():
    """Get the path to python in the venv"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def install_dependencies(gpu_platform: str = "cuda"):
    """Install dependencies for the chosen GPU backend."""
    python_path = get_python_path()

    if not python_path.exists():
        print(f"Error: python not found at {python_path}")
        return False

    print("Upgrading pip...")
    subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)

    print("Installing uv...")
    subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "uv"], check=True)

    if gpu_platform == "rocm":
        if platform.system() == "Windows":
            print("ERROR: Windows AMD ROCm is not supported by this script.")
            print("Use install_fizgig_rocm.bat instead — it installs up-to-date ROCm nightly wheels.")
            return False
        if not REQUIREMENTS_GLOBAL.exists() or not REQUIREMENTS_ROCM_LINUX.exists():
            print("ERROR: requirements-global.txt or requirements-rocm-linux.txt is missing.")
            return False
        req_files = [REQUIREMENTS_GLOBAL, REQUIREMENTS_ROCM_LINUX]
        print("Installing ROCm PyTorch + shared dependencies (Linux)...")
    else:
        req_files = [REQUIREMENTS_FILE]
        print(f"Installing dependencies from: {REQUIREMENTS_FILE} (CUDA, using uv)")

    print("(This may take a few minutes for PyTorch download...)")

    try:
        cmd = [str(python_path), "-m", "uv", "pip", "install", "--link-mode", "copy",
               "--index-strategy", "unsafe-best-match"]
        for req in req_files:
            cmd.extend(["-r", str(req)])
        subprocess.run(cmd, check=True)
        print("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False


def verify_gpu():
    """Probe torch.cuda after install (CUDA or ROCm/HIP backend)."""
    python_path = get_python_path()
    probe_script = '''
import sys
try:
    import torch
except Exception as e:
    print(f"FAIL: torch import failed: {e}")
    sys.exit(2)

if torch.cuda.is_available():
    try:
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        hip = getattr(torch.version, "hip", None)
        rocm = getattr(torch.version, "rocm", None)
        backend = "ROCm/HIP" if (hip or rocm or "+rocm" in torch.__version__.lower()) else "CUDA"
        runtime = rocm or hip or getattr(torch.version, "cuda", None) or "unknown"
        print(f"OK  {backend} visible: {name} ({vram_gb:.1f} GB VRAM)")
        print(f"    PyTorch {torch.__version__}  |  runtime {runtime}")
        sys.exit(0)
    except Exception as e:
        print(f"WARN GPU reported available but device query failed: {e}")
        sys.exit(1)
else:
    print("WARN PyTorch installed but torch.cuda.is_available() is False.")
    print("     Fizgig training (Klein 9B / Krea 2) needs a CUDA- or ROCm-capable GPU.")
    print("     NVIDIA: update driver to 555+ (Windows) / 550+ (Linux) and re-run.")
    print("     AMD Windows: use install_fizgig_rocm.bat for ROCm nightly wheels.")
    sys.exit(1)
'''
    try:
        result = subprocess.run(
            [str(python_path), "-c", probe_script],
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        return result.returncode == 0
    except Exception as e:
        print(f"Note: GPU probe skipped: {e}")
        return True  # Don't fail installation


# Backward-compatible alias
verify_cuda = verify_gpu


def download_insightface_models():
    """Trigger InsightFace model download by running a test detection"""
    python_path = get_python_path()

    print("Downloading InsightFace models (buffalo_l)...")
    print("(First-time download, ~300MB)")

    # Python script to download models
    download_script = '''
import os
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from insightface.app import FaceAnalysis

    print("Initializing FaceAnalysis (this downloads models)...")
    app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection', 'genderage', 'recognition'],
        providers=['CPUExecutionProvider']
    )
    app.prepare(ctx_id=-1)  # -1 = CPU only
    print("Models downloaded successfully!")
    sys.exit(0)
except Exception as e:
    print(f"Warning: Could not download models: {e}")
    print("Models will be downloaded on first use.")
    sys.exit(0)  # Don't fail installation
'''

    try:
        result = subprocess.run(
            [str(python_path), "-c", download_script],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            # Filter out common warnings
            for line in result.stderr.split('\n'):
                if line and 'WARNING' not in line and 'deprecated' not in line.lower():
                    print(line)
        return True
    except Exception as e:
        print(f"Note: Model pre-download skipped: {e}")
        print("Models will download automatically on first use.")
        return True  # Don't fail installation


def verify_rocm_vram():
    """Linux ROCm: probe amd-smi/rocm-smi for status-bar VRAM reads (informational)."""
    if platform.system() != "Linux":
        return True
    python_path = get_python_path()
    probe_script = '''
import sys
sys.path.insert(0, "src")
try:
    from fizgig.utils.vram_monitor import _read_vram_amd_smi_cli, _read_vram_rocm_smi
    hit = _read_vram_amd_smi_cli() or _read_vram_rocm_smi()
    if hit:
        used, total = hit
        print(f"OK  VRAM monitor: {used / (1024**3):.1f} / {total / (1024**3):.1f} GB in use")
        sys.exit(0)
    print("WARN VRAM monitor: amd-smi and rocm-smi unavailable — status bar may show")
    print("     allocator-only usage. Install amdrocm-amdsmi (see requirements-rocm-linux.txt).")
    sys.exit(1)
except Exception as e:
    print(f"WARN VRAM monitor probe failed: {e}")
    sys.exit(1)
'''
    try:
        result = subprocess.run(
            [str(python_path), "-c", probe_script],
            capture_output=True, text=True,
            cwd=str(SCRIPT_DIR),
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        return result.returncode == 0
    except Exception as e:
        print(f"Note: VRAM monitor probe skipped: {e}")
        return True


def create_launcher_scripts(gpu_platform: str = "cuda"):
    """Verify/create launcher scripts.

    Windows: run_fizgig.bat ships WITH the repo (the consoleless chain: .bat -> run_silent.vbs
    -> pythonw launch.pyw). This step used to overwrite it with an ancient console-attached
    version (python + pause), which (a) gave every fresh install a lingering console window
    that ended in 'Press any key to continue', and (b) dirtied a tracked file so the next
    update_fizgig.bat's `git pull` refused to run. Never write it here.
    """
    bat_path = SCRIPT_DIR / "run_fizgig.bat"
    if bat_path.exists():
        print(f"Launcher present: {bat_path} (ships with the repo — not modified)")
    else:
        print(f"WARNING: {bat_path} is missing — restore it with `git checkout -- run_fizgig.bat`")

    # Linux/Mac shell script (not shipped in the repo — generated here)
    rocm_env = ""
    if gpu_platform == "rocm" or (platform.system() == "Linux" and os.path.exists("/dev/kfd")):
        rocm_env = """\
export MIOPEN_FIND_MODE=2
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
export FIZGIG_GPU_BACKEND=rocm
for _d in /opt/rocm/core-*/bin /opt/rocm/bin; do
  [ -d "$_d" ] && PATH="$_d:$PATH"
done
export PATH
if [ -d /opt/rocm/lib ]; then
  export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
"""

    sh_content = f'''#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
{rocm_env}python lora_trainer_gui.py
'''

    sh_path = SCRIPT_DIR / "run_fizgig.sh"
    existed = sh_path.exists()
    if not existed or gpu_platform == "rocm":
        with open(sh_path, 'w', newline='\n') as f:
            f.write(sh_content)
        if platform.system() != "Windows":
            os.chmod(sh_path, 0o755)
        print(f"{'Updated' if existed else 'Created'}: {sh_path}")
    return True


def print_summary(gpu_platform: str = "cuda"):
    """Print installation summary and next steps"""
    print_header("Installation Complete!")

    print("\nFizgig has been installed successfully.")
    print("\nTo launch Fizgig:")

    if platform.system() == "Windows":
        if gpu_platform == "rocm":
            print(f"  Double-click: {SCRIPT_DIR / 'run_fizgig_rocm.bat'}")
            print("  Or run: .\\run_fizgig_rocm.bat")
        else:
            print(f"  Double-click: {SCRIPT_DIR / 'run_fizgig.bat'}")
            print("  Or run: .\\run_fizgig.bat")
    else:
        print(f"  Run: ./run_fizgig.sh")

    backend = "ROCm/HIP" if gpu_platform == "rocm" else "CUDA"
    print(f"\nGPU backend: {backend}")

    print("\nFace Detection:")
    print("  - InsightFace models will download on first use (~300MB)")
    print("  - Uses CPU for detection (works on any GPU)")
    print("  - Supports gender classification for face filtering")

    print("\nAI Captioning (Florence-2):")
    print("  - Florence models auto-download from Hugging Face on first use")
    print("  - Model sizes: base (~500MB), large (~1.5GB)")
    print("  - Runs on GPU via PyTorch (CUDA or ROCm)")

    print("\n" + "=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Install Fizgig and its dependencies.")
    parser.add_argument(
        "--platform",
        choices=("detect", "cuda", "rocm"),
        default="detect",
        help="GPU backend to install (default: auto-detect)",
    )
    return parser.parse_args()


def check_msvc_build_tools():
    """Windows: detect the MSVC C++ Build Tools torch.compile's inductor needs.

    triton installs via requirements.txt, but inductor also compiles host-side C++ —
    without MSVC, Compile Blocks silently runs eager (a console note at training time).
    The MS download site is ambiguous, so print the EXACT installer link and the one
    workload to tick. Informational only; training works fine without."""
    if os.name != "nt":
        print("  Not Windows — torch.compile uses the system toolchain, nothing to check.")
        return
    import glob as _glob
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    found = False
    if os.path.exists(vswhere):
        try:
            out = subprocess.run(
                [vswhere, "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, timeout=30)
            found = bool(out.stdout.strip())
        except Exception:
            pass
    if not found:
        # Fallback sweep for cl.exe in the standard install roots (covers odd setups).
        for root in (r"C:\Program Files\Microsoft Visual Studio",
                     r"C:\Program Files (x86)\Microsoft Visual Studio"):
            if _glob.glob(os.path.join(root, "*", "*", "VC", "Tools", "MSVC", "*",
                                       "bin", "Hostx64", "x64", "cl.exe")):
                found = True
                break
    if found:
        print("  MSVC C++ Build Tools found — torch.compile speedups are available.")
        return
    print("  OPTIONAL: MSVC C++ Build Tools not detected.")
    print("  The torch.compile training speedup needs them; everything else works without.")
    print()
    print("  Direct download (exact installer, no hunting on the MS site):")
    print("    https://aka.ms/vs/17/release/vs_BuildTools.exe")
    print('  During install, tick the "Desktop development with C++" workload.')
    print()
    print("  Or install unattended from a terminal:")
    print('    winget install Microsoft.VisualStudio.2022.BuildTools --override '
          '"--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"')
    print()
    print("  Install it any time — no need to re-run this installer. The next training run")
    print("  detects it automatically.")


def main():
    args = parse_args()
    gpu_platform = resolve_platform(args.platform)

    print_header("Fizgig Installer — Klein 9B & Krea 2 LoRA Workbench")
    print(f"Installation directory: {SCRIPT_DIR}")
    if gpu_platform == "rocm":
        print("GPU backend: AMD ROCm")
    else:
        print("GPU backend: NVIDIA CUDA 12.8")

    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    if not check_python_version():
        sys.exit(1)

    # Step 2: Create virtual environment
    print_step(2, "Creating virtual environment")
    if not create_venv():
        sys.exit(1)

    # Step 3: Install dependencies
    print_step(3, f"Installing dependencies ({gpu_platform.upper()})")
    if not install_dependencies(gpu_platform):
        sys.exit(1)

    # Step 4: Verify GPU is visible to PyTorch
    print_step(4, "Verifying GPU availability")
    verify_gpu()  # Don't fail on this — user may be on a CPU-only machine for prep only

    if gpu_platform == "rocm" and platform.system() == "Linux":
        print("\n[ROCm] Checking VRAM monitor (amd-smi / rocm-smi)...")
        print("-" * 40)
        verify_rocm_vram()

    # Step 5: Download InsightFace models
    print_step(5, "Downloading face detection models")
    download_insightface_models()  # Don't fail on this

    # Step 6: Create launcher scripts
    print_step(6, "Creating launcher scripts")
    if not create_launcher_scripts(gpu_platform):
        sys.exit(1)

    # Step 7: MSVC C++ Build Tools check (torch.compile's inductor backend needs them on
    # Windows; triton itself comes from requirements.txt). Informational — never fails.
    print_step(7, "Checking for MSVC C++ Build Tools (torch.compile)")
    check_msvc_build_tools()

    # Print summary
    print_summary(gpu_platform)


if __name__ == "__main__":
    main()
