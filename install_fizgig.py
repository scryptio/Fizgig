#!/usr/bin/env python3
"""
Fizgig Installer
================
Sets up Fizgig with its own Python virtual environment.
Installs PyTorch built against CUDA 12.8 — required for Klein 9B and Krea 2 training,
preview rendering, profiling, extraction, and Repair Studio on NVIDIA GPUs.

Features:
- Creates isolated venv for Fizgig dependencies
- Installs CUDA 12.8 PyTorch (supports RTX 30xx / 40xx / 50xx Blackwell)
- Installs InsightFace face detection (runs on CPU for GPU independence)
- Downloads face detection models automatically
- Installs Florence-2 AI captioning (transformers library; runs on GPU)
- Creates launcher scripts for Windows/Linux/Mac
- Verifies CUDA is visible to PyTorch after install

Note: Florence-2 models are auto-downloaded from Hugging Face on first use (~500MB-1.5GB)
"""

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


def install_dependencies():
    """Install dependencies from requirements.txt"""
    python_path = get_python_path()

    if not python_path.exists():
        print(f"Error: python not found at {python_path}")
        return False

    print("Upgrading pip...")
    subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)

    print("Installing uv...")
    # shell=False (list form) with internal Path constants — not injectable
    subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "uv"], check=True)

    print(f"Installing dependencies from: {REQUIREMENTS_FILE} (using uv)")
    print("(This may take a few minutes for PyTorch download...)")

    from uv_install_deps import install_requirements
    if install_requirements(REQUIREMENTS_FILE, VENV_DIR, python_path):
        print("Dependencies installed successfully.")
        return True
    return False


def verify_cuda():
    """Probe torch.cuda after install so users learn about driver mismatches
    at install time rather than mid-training."""
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
        print(f"OK  CUDA visible: {name} ({vram_gb:.1f} GB VRAM)")
        print(f"    PyTorch {torch.__version__}  |  CUDA runtime {torch.version.cuda}")
        sys.exit(0)
    except Exception as e:
        print(f"WARN CUDA reported available but device query failed: {e}")
        sys.exit(1)
else:
    print("WARN PyTorch installed but torch.cuda.is_available() is False.")
    print("     Fizgig training (Klein 9B / Krea 2) needs a CUDA-capable GPU.")
    print("     If you have one, update your NVIDIA driver to 555+ and re-run this installer.")
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
        print(f"Note: CUDA probe skipped: {e}")
        return True  # Don't fail installation


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


def create_launcher_scripts():
    """Verify/create launcher scripts.

    Windows: run_fizgig.bat ships WITH the repo (the consoleless chain: .bat -> run_silent.vbs
    -> pythonw launch.pyw). This step used to overwrite it with an ancient console-attached
    version (python + pause), which (a) gave every fresh install a lingering console window
    that ended in 'Press any key to continue', and (b) dirtied a tracked file so the next
    update_fizgig.bat's `git pull` refused to run. Never write it here.
    """
    # Gizmo's launcher is the same story and gets the same treatment: tracked, verified, never
    # written. It needs no install of its own — Tkinter is stdlib, PIL and imageio-ffmpeg are
    # already pinned — so it simply arrives with an ordinary update.
    for name in ("run_fizgig.bat", "Launch Gizmo (Video clip prep tool).bat"):
        bat_path = SCRIPT_DIR / name
        if bat_path.exists():
            print(f"Launcher present: {bat_path} (ships with the repo — not modified)")
        else:
            print(f"WARNING: {bat_path} is missing — restore it with "
                  f'`git checkout -- "{name}"`')

    # Linux/Mac shell script (not shipped in the repo — generated here)
    sh_content = '''#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python lora_trainer_gui.py
'''

    sh_path = SCRIPT_DIR / "run_fizgig.sh"
    if not sh_path.exists():
        with open(sh_path, 'w', newline='\n') as f:
            f.write(sh_content)
        if platform.system() != "Windows":
            os.chmod(sh_path, 0o755)
        print(f"Created: {sh_path}")
    return True


def print_summary():
    """Print installation summary and next steps"""
    print_header("Installation Complete!")

    print("\nFizgig has been installed successfully.")
    print("\nTo launch Fizgig:")

    if platform.system() == "Windows":
        print(f"  Double-click: {SCRIPT_DIR / 'run_fizgig.bat'}")
        print("  Or run: .\\run_fizgig.bat")
    else:
        print(f"  Run: ./run_fizgig.sh")

    print("\nFace Detection:")
    print("  - InsightFace models will download on first use (~300MB)")
    print("  - Uses CPU for detection (works on any GPU)")
    print("  - Supports gender classification for face filtering")

    print("\nAI Captioning (Florence-2):")
    print("  - Florence models auto-download from Hugging Face on first use")
    print("  - Model sizes: base (~500MB), large (~1.5GB)")
    print("  - Runs on GPU via CUDA 12.8 PyTorch (fast)")

    print("\n" + "=" * 60)


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


def _windows_has_amd_radeon() -> bool:
    """Best-effort AMD detect for the hand-off message (PowerShell CIM)."""
    ps = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object -ExpandProperty Name"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            text = result.stdout or ""
            return "AMD" in text and "Radeon" in text
    except Exception:
        pass
    return False


def redirect_amd_to_rocm_installer():
    """AMD support is additive: leave the CUDA path untouched and hand off early."""
    if platform.system() == "Linux":
        has_nvidia = os.path.exists("/dev/nvidia0") or bool(shutil.which("nvidia-smi"))
        if os.path.exists("/dev/kfd") and not has_nvidia:
            print("AMD ROCm detected. This installer is NVIDIA CUDA only.")
            print("Use the separate AMD installer (Linux AMD is highly experimental):")
            print(f"  chmod +x {SCRIPT_DIR / 'install_fizgig_rocm.sh'}")
            print(f"  {SCRIPT_DIR / 'install_fizgig_rocm.sh'}")
            sys.exit(1)
        return

    if platform.system() == "Windows":
        if shutil.which("nvidia-smi"):
            return
        if _windows_has_amd_radeon():
            print("AMD GPU detected. This installer is NVIDIA CUDA only.")
            print("Use the separate AMD installer:")
            print(f"  {SCRIPT_DIR / 'install_fizgig_rocm.bat'}")
            sys.exit(1)


def main():
    print_header("Fizgig Installer — Klein 9B & Krea 2 LoRA Workbench")
    print(f"Installation directory: {SCRIPT_DIR}")

    redirect_amd_to_rocm_installer()

    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    if not check_python_version():
        sys.exit(1)

    # Step 2: Create virtual environment
    print_step(2, "Creating virtual environment")
    if not create_venv():
        sys.exit(1)

    # Step 3: Install dependencies
    print_step(3, "Installing dependencies")
    if not install_dependencies():
        sys.exit(1)

    # Step 4: Verify CUDA is visible to PyTorch
    print_step(4, "Verifying CUDA availability")
    verify_cuda()  # Don't fail on this — user may be on a CPU-only machine for prep only

    # Step 5: Download InsightFace models
    print_step(5, "Downloading face detection models")
    download_insightface_models()  # Don't fail on this

    # Step 6: Create launcher scripts
    print_step(6, "Creating launcher scripts")
    if not create_launcher_scripts():
        sys.exit(1)

    # Step 7: MSVC C++ Build Tools check (torch.compile's inductor backend needs them on
    # Windows; triton itself comes from requirements.txt). Informational — never fails.
    print_step(7, "Checking for MSVC C++ Build Tools (torch.compile)")
    check_msvc_build_tools()

    # Print summary
    print_summary()


if __name__ == "__main__":
    main()
