#!/usr/bin/env bash
# Fizgig ROCm Linux installer — AMD pip wheels (Path B).
# HIGHLY EXPERIMENTAL: Linux AMD training is best-effort only (driver resets, gfx gaps,
# desktop+compute contention). Use Windows ROCm or NVIDIA Linux for production workloads.
# Detects gfx target, installs PyTorch/ROCm from AMD multi-arch wheels into venv,
# then Fizgig deps from requirements-global.txt.
# Prerequisites: amdgpu driver, /dev/kfd, user in render/video groups; sudo for libnuma-dev / pythonX.Y-dev.
set -euo pipefail

FIZGIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROCM_INDEX="${ROCM_INDEX:-https://repo.amd.com/rocm/whl-multi-arch/}"
ROCM_NIGHTLY_INDEX="${ROCM_NIGHTLY_INDEX:-https://rocm.nightlies.amd.com/whl-multi-arch/}"
# stable = ROCm 7.14 from repo.amd.com (matches libbitsandbytes_rocm714.so / BNB_ROCM_VERSION=714).
# nightly = newer torch builds from rocm.nightlies.amd.com, still pinned to ROCm 7.14 for bitsandbytes.
ROCM_CHANNEL="${ROCM_CHANNEL:-stable}"
ROCM_SDK_PIN="${ROCM_SDK_PIN:-7.14.0}"
TORCH_PIN="${TORCH_PIN:-2.13.0+rocm7.14.0}"
# Set CLEAR_PIP_CACHE=1 to run `pip cache purge` and pass --no-cache-dir for torch/vision wheels.
CLEAR_PIP_CACHE="${CLEAR_PIP_CACHE:-0}"

# python3 used to create venv — not conda; override with FIZGIG_PYTHON=/usr/bin/python3.12
_fizgig_install_python() {
    if [[ -n "${FIZGIG_PYTHON:-}" ]]; then
        echo "$FIZGIG_PYTHON"
        return 0
    fi
    if [[ -n "${CONDA_DEFAULT_ENV:-}" ]] || [[ -n "${CONDA_PREFIX:-}" ]]; then
        echo "ERROR: Conda environment is active (${CONDA_DEFAULT_ENV:-unknown})." >&2
        echo "       Fizgig builds a local venv from system python3 — deactivate conda first:" >&2
        echo "         conda deactivate" >&2
        exit 1
    fi
    local py_path py
    py_path="$(command -v python3)"
    if [[ "$py_path" == *"/conda/"* ]] || [[ "$py_path" == *"/miniconda"* ]] || [[ "$py_path" == *"/anaconda"* ]]; then
        if [[ -x /usr/bin/python3 ]]; then
            local sys_py
            sys_py="$(readlink -f /usr/bin/python3 2>/dev/null || echo /usr/bin/python3)"
            if [[ "$sys_py" != *conda* ]] && [[ "$sys_py" != *miniconda* ]] && [[ "$sys_py" != *anaconda* ]]; then
                echo "NOTE: python3 on PATH is conda; using /usr/bin/python3 for the Fizgig venv." >&2
                py="/usr/bin/python3"
            else
                echo "ERROR: python3 and /usr/bin/python3 both appear to be conda." >&2
                echo "       Deactivate conda or set FIZGIG_PYTHON to a system interpreter." >&2
                exit 1
            fi
        else
            echo "ERROR: python3 is conda and /usr/bin/python3 was not found." >&2
            exit 1
        fi
    else
        py="python3"
    fi
    echo "$py"
}

_ensure_python_dev_headers() {
    local py="$1"
    local py_mm header
    py_mm="$("$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    header="/usr/include/python${py_mm}/Python.h"
    if [[ -f "$header" ]]; then
        return 0
    fi
    echo "Installing python${py_mm}-dev (headers for $($py --version | cut -d' ' -f2) — Triton/torch.compile needs Python.h)..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get install -y "python${py_mm}-dev" || sudo apt-get install -y python3-dev
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y "python${py_mm}-devel" || sudo dnf install -y python3-devel
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y "python${py_mm}-devel" || sudo yum install -y python3-devel
    else
        echo "ERROR: Python.h not found (${header}). Install python${py_mm}-dev, then re-run."
        exit 1
    fi
    if [[ ! -f "$header" ]]; then
        echo "ERROR: Python.h still missing at ${header} after installing python${py_mm}-dev." >&2
        exit 1
    fi
}

_clear_pip_cache() {
    local py
    py="$(_fizgig_rocm_python)"
    echo "Clearing pip wheel cache (venv pip → ~/.cache/pip)..."
    "$py" -m pip cache purge 2>/dev/null || rm -rf "${HOME}/.cache/pip" 2>/dev/null || true
}

_purge_rocm_torch() {
    local py
    py="$(_fizgig_rocm_python)"
    local pkg
    while IFS= read -r pkg; do
        [[ -n "$pkg" ]] && "$py" -m pip uninstall -y "$pkg" 2>/dev/null || true
    done < <("$py" - <<'PY'
import importlib.metadata as md

for dist in md.distributions():
    name = (dist.metadata.get("Name") or "").lower()
    if name.startswith(("torch", "rocm", "amd-torch", "amd-torchvision", "triton")):
        print(dist.metadata["Name"])
PY
)
}

_fizgig_rocm_python() {
    if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
        echo "${VIRTUAL_ENV}/bin/python"
    elif [[ -x "${FIZGIG_ROOT}/venv/bin/python" ]]; then
        echo "${FIZGIG_ROOT}/venv/bin/python"
    else
        echo python3
    fi
}

# Nightly: resolve latest installable torch +rocm7.14; stable uses _resolve_stable_stack instead.
_resolve_torch714() {
    local index="$1"
    local py
    py="$(_fizgig_rocm_python)"
    TORCH_VER=""
    if [[ -n "${TORCH_PIN:-}" ]]; then
        TORCH_VER="$TORCH_PIN"
        return 0
    fi
    while IFS= read -r line; do
        case "$line" in
            TORCH_VER=*) TORCH_VER="${line#TORCH_VER=}" ;;
        esac
    done < <("$py" - <<PY
import json
import re
import subprocess
import sys

index = """${index}"""
pin = """${ROCM_SDK_PIN}"""


def semver_tuple(v: str) -> tuple[int, ...]:
    base = v.split("+", 1)[0]
    return tuple(int(p) for p in re.split(r"[.\-]", base) if p.isdigit())


def pip_versions(package: str) -> list[str]:
    proc = subprocess.run(
        [
            sys.executable, "-m", "pip", "index", "versions", package,
            "--index-url", index, "--pre", "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout, file=sys.stderr)
        sys.exit(1)
    return json.loads(proc.stdout)["versions"]


def rocm714_torch() -> list[str]:
    return [
        v for v in pip_versions("torch")
        if re.search(r"\+rocm7\.14", v, re.I) and not re.search(r"\+rocm7\.15", v, re.I)
    ]


def rocm_req(v: str) -> str | None:
    m = re.search(r"\+rocm(7\.14\.0(?:a\d+)?)", v, re.I)
    return m.group(1) if m else None


def sort_key(v: str) -> tuple[tuple[int, ...], int, int]:
    semver = semver_tuple(v)
    m = re.search(r"\+rocm7\.14\.0a(\d+)", v, re.I)
    if m:
        return semver, int(m.group(1)), 1
    if re.search(r"\+rocm7\.14", v, re.I):
        return semver, 0, 0
    return semver, -1, -1


rocm714_meta = {v for v in pip_versions("rocm") if v == pin or v.startswith(f"{pin}a")}
cands = [v for v in rocm714_torch() if (req := rocm_req(v)) and req in rocm714_meta]
if not cands:
    # stable index may use release rocm==7.14.0 without alpha builds in torch tag
    cands = rocm714_torch()
if not cands:
    print(f"ERROR: no torch +rocm7.14 on {index}", file=sys.stderr)
    sys.exit(1)

print(f"TORCH_VER={max(cands, key=sort_key)}")
PY
)
    if [[ -z "$TORCH_VER" ]]; then
        echo "ERROR: failed to resolve torch +rocm7.14 from ${index}" >&2
        return 1
    fi
}

# Stable: never use torch[device-*] extras (2.13+ warns and skips device wheels). Install
# amd-torch-device-{arch} explicitly when that package exists on the index.
_resolve_stable_stack() {
    local index="$1"
    local py
    py="$(_fizgig_rocm_python)"
    TORCH_VER=""
    VISION_VER=""
    STABLE_DEVICE_WHEEL=0
    if [[ -n "${TORCH_PIN:-}" ]]; then
        TORCH_VER="$TORCH_PIN"
    fi
    while IFS= read -r line; do
        case "$line" in
            TORCH_VER=*) TORCH_VER="${line#TORCH_VER=}" ;;
            VISION_VER=*) VISION_VER="${line#VISION_VER=}" ;;
            STABLE_DEVICE_WHEEL=*) STABLE_DEVICE_WHEEL="${line#STABLE_DEVICE_WHEEL=}" ;;
        esac
    done < <("$py" - <<PY
import json
import re
import subprocess
import sys

index = """${index}"""
arch = """${ARCH}"""
torch_pin = """${TORCH_PIN:-}"""
torch_device_pkg = f"amd-torch-device-{arch}"
vision_device_pkg = f"amd-torchvision-device-{arch}"


def semver_tuple(v: str) -> tuple[int, ...]:
    base = v.split("+", 1)[0]
    return tuple(int(p) for p in re.split(r"[.\-]", base) if p.isdigit())


def pip_versions(package: str) -> list[str]:
    proc = subprocess.run(
        [
            sys.executable, "-m", "pip", "index", "versions", package,
            "--index-url", index, "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return json.loads(proc.stdout)["versions"]


def rocm714_only(versions: list[str]) -> list[str]:
    return [
        v for v in versions
        if re.search(r"\+rocm7\.14", v, re.I) and not re.search(r"\+rocm7\.15", v, re.I)
    ]


def vision_for_torch(torch_ver: str) -> str:
    base = torch_ver.split("+", 1)[0]
    rocm_tag = torch_ver.split("+", 1)[1] if "+" in torch_ver else ""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)(.*)$", base)
    if not m:
        print(f"ERROR: cannot parse torch version {torch_ver!r}", file=sys.stderr)
        sys.exit(1)
    _major, minor, patch, rest = m.groups()
    vision_base = f"0.{int(minor) + 15}.{patch}{rest}"
    exact = f"{vision_base}+{rocm_tag}" if rocm_tag else vision_base
    versions = rocm714_only(pip_versions("torchvision"))
    if exact in versions:
        return exact
    prefix = f"{vision_base}+"
    cands = [v for v in versions if v.startswith(prefix)]
    if cands:
        return max(cands, key=semver_tuple)
    print(f"ERROR: no torchvision match for torch {torch_ver} on {index}", file=sys.stderr)
    sys.exit(1)


torch_device_vers = rocm714_only(pip_versions(torch_device_pkg))
vision_device_vers = rocm714_only(pip_versions(vision_device_pkg))
if not torch_device_vers or not vision_device_vers:
    if torch_pin:
        print(f"TORCH_VER={torch_pin}")
        print(f"VISION_VER={vision_for_torch(torch_pin)}")
    print("STABLE_DEVICE_WHEEL=0")
    sys.exit(0)

torch_ver = torch_pin if torch_pin else max(torch_device_vers, key=semver_tuple)
vision_ver = vision_for_torch(torch_ver)
vision_prefix = vision_ver.split("+", 1)[0] + "+"
if vision_ver not in vision_device_vers and not any(v.startswith(vision_prefix) for v in vision_device_vers):
    if torch_pin:
        print(f"TORCH_VER={torch_pin}")
        print(f"VISION_VER={vision_ver}")
    print("STABLE_DEVICE_WHEEL=0")
    sys.exit(0)

print(f"TORCH_VER={torch_ver}")
print(f"VISION_VER={vision_ver}")
print("STABLE_DEVICE_WHEEL=1")
PY
)
}

verify_torch_device_wheel() {
    local py
    py="$(_fizgig_rocm_python)"
    if [[ "${STABLE_DEVICE_WHEEL:-0}" != "1" ]]; then
        return 0
    fi
    "$py" - <<PY
import importlib.metadata as md
import sys

import torch
import torchvision

arch = "${ARCH}"
checks = [
    (f"amd-torch-device-{arch}", torch.__version__),
    (f"amd-torchvision-device-{arch}", torchvision.__version__),
]
for pkg_prefix, want_ver in checks:
    needle = pkg_prefix.lower()
    found = []
    for d in md.distributions():
        name = (d.metadata.get("Name") or "").lower()
        if name == needle:
            found.append(d.metadata.get("Name", ""))
            if d.version != want_ver:
                print(
                    f"ERROR: {d.metadata.get('Name')} {d.version} != expected {want_ver}",
                    file=sys.stderr,
                )
                sys.exit(1)
    if not found:
        print(f"ERROR: missing {pkg_prefix}=={want_ver}", file=sys.stderr)
        sys.exit(1)
    print(f"OK  {found[0]} (matches {want_ver})")
PY
}

install_rocm_torch_wheels() {
    : "${ARCH:?ARCH must be set before installing torch}"
    local index="$1"
    shift
    local -a pip_extra=("$@")
    local -a pip_pkgs=()
    local py
    py="$(_fizgig_rocm_python)"

    local -a pip_pre=()
    local -a pip_upgrade=(--force-reinstall)
    STABLE_DEVICE_WHEEL=0

    _purge_rocm_torch
    if [[ "$CLEAR_PIP_CACHE" == "1" ]]; then
        pip_extra+=(--no-cache-dir)
    fi

    if [[ "$index" == *nightlies* ]]; then
        _resolve_torch714 "$index"
        pip_pkgs=(
            "torch[device-${ARCH}]==${TORCH_VER}"
            "torchvision[device-${ARCH}]"
            "rocm-sdk-devel"
        )
        pip_pre=(--pre)
        pip_upgrade=(--upgrade --force-reinstall)
        STABLE_DEVICE_WHEEL=1
        echo "Installing torch[device-${ARCH}]==${TORCH_VER} + torchvision[device-${ARCH}] + rocm-sdk-devel (nightly)..."
    else
        _resolve_stable_stack "$index"
        if [[ "${STABLE_DEVICE_WHEEL}" == "1" ]]; then
            pip_pkgs=(
                "torch==${TORCH_VER}"
                "torchvision==${VISION_VER}"
                "amd-torch-device-${ARCH}==${TORCH_VER}"
                "amd-torchvision-device-${ARCH}==${VISION_VER}"
                "rocm-sdk-devel==${ROCM_SDK_PIN}"
            )
            echo "Installing torch==${TORCH_VER} + torchvision==${VISION_VER} + amd-torch-device-${ARCH} + amd-torchvision-device-${ARCH} + rocm-sdk-devel==${ROCM_SDK_PIN} ..."
        elif [[ -n "${TORCH_VER:-}" ]]; then
            pip_pkgs=(
                "torch==${TORCH_VER}"
                "torchvision==${VISION_VER}"
                "rocm-sdk-devel==${ROCM_SDK_PIN}"
            )
            echo "Installing torch==${TORCH_VER} + torchvision==${VISION_VER} + rocm-sdk-devel==${ROCM_SDK_PIN} (stable — TORCH_PIN, no device wheel)..."
        else
            pip_pkgs=(
                "torch"
                "torchvision"
                "rocm-sdk-devel==${ROCM_SDK_PIN}"
            )
            echo "Installing torch + torchvision + rocm-sdk-devel==${ROCM_SDK_PIN} (stable, unpinned)..."
        fi
    fi

    "$py" -m pip install "${pip_pre[@]}" "${pip_upgrade[@]}" --index-url "$index" \
        "${pip_pkgs[@]}" \
        "${pip_extra[@]}"
}

verify_torch_rocm_pin() {
    local py
    py="$(_fizgig_rocm_python)"
    "$py" - <<PY
import re
import sys

import torch

pin = "${ROCM_SDK_PIN}"
want = ".".join(pin.split(".")[:2])
got = None

rocm = getattr(torch.version, "rocm", None)
if rocm:
    m = re.match(r"(\d+\.\d+)", str(rocm))
    if m:
        got = m.group(1)

if got is None:
    m = re.search(r"\+rocm(\d+\.\d+)", torch.__version__, re.I)
    if m:
        got = m.group(1)

if got != want:
    print(
        f"ERROR: PyTorch ROCm {got or '?'} != required {want} "
        f"(bitsandbytes needs libbitsandbytes_rocm714.so / BNB_ROCM_VERSION=714)",
        file=sys.stderr,
    )
    print(f"       torch {torch.__version__}", file=sys.stderr)
    sys.exit(1)

print(f"OK  torch ROCm {got} matches bitsandbytes (BNB_ROCM_VERSION=714)")
PY
}

verify_torch_gpu_kernel() {
    local py
    py="$(_fizgig_rocm_python)"
    "$py" - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    print("WARN GPU kernel test skipped (cuda not available)", file=sys.stderr)
    sys.exit(1)
try:
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    y = x @ x
    torch.cuda.synchronize()
    del x, y
except Exception as exc:
    print(f"ERROR: GPU kernel smoke test failed: {exc}", file=sys.stderr)
    sys.exit(1)
print("OK  GPU kernel smoke test passed")
PY
}

verify_torchvision() {
    local py
    py="$(_fizgig_rocm_python)"
    "$py" - <<'PY'
import sys

import torch
import torchvision

print(f"torch {torch.__version__}  torchvision {torchvision.__version__}")
try:
    from torchvision.transforms import InterpolationMode  # noqa: F401
except Exception as exc:
    print(f"ERROR: torchvision incompatible with torch: {exc}", file=sys.stderr)
    sys.exit(1)
print("OK  torchvision import (torch/torchvision ABI match)")
PY
}

install_rocm_torch_pinned() {
    local index="$1"
    shift

    install_rocm_torch_wheels "$index" "$@" || return 1
    verify_torch_rocm_pin || return 1
    verify_torch_device_wheel || return 1
    verify_torch_gpu_kernel || return 1
    verify_torchvision || return 1
    return 0
}

install_torch_stable() {
    install_rocm_torch_pinned "$ROCM_INDEX" || return 1
}

install_torch_nightly() {
    install_rocm_torch_pinned "$ROCM_NIGHTLY_INDEX" || return 1
}

cd "$FIZGIG_ROOT"

echo "============================================================"
echo "  Fizgig Installer — AMD ROCm (Linux, pip wheels)"
echo "  Klein 9B and Krea 2 LoRA Studio"
echo "  *** HIGHLY EXPERIMENTAL — Linux AMD is best-effort only ***"
echo "============================================================"
echo

if [[ ! -e /dev/kfd ]]; then
    echo "ERROR: /dev/kfd not found — AMD ROCm kernel driver is not loaded."
    echo "Install the amdgpu/ROCm stack first (see AMD ROCm install docs), then re-run."
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found."
    exit 1
fi

INSTALL_PYTHON="$(_fizgig_install_python)"

"$INSTALL_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "ERROR: Python 3.10+ required."
    exit 1
}

echo "Python: $("$INSTALL_PYTHON" --version) ($("$INSTALL_PYTHON" -c 'import sys; print(sys.executable)'))"
echo

_ensure_python_dev_headers "$INSTALL_PYTHON"

if [[ ! -e /usr/lib/x86_64-linux-gnu/libnuma.so ]] && [[ ! -e /lib/x86_64-linux-gnu/libnuma.so ]]; then
    echo "Installing libnuma-dev (PyTorch rocSHMEM needs unversioned libnuma.so)..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get install -y libnuma-dev
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y numactl-devel
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y numactl-devel
    else
        echo "ERROR: libnuma.so not found and no supported package manager (apt/dnf/yum)."
        echo "Install libnuma-dev (Debian/Ubuntu) or numactl-devel (Fedora/RHEL), then re-run."
        exit 1
    fi
fi

if [[ -d venv ]]; then
    read -r -p "Virtual environment already exists at venv/. Delete and recreate? (y/N): " RECREATE
    if [[ "${RECREATE,,}" == "y" ]]; then
        rm -rf venv
    fi
fi

if [[ ! -d venv ]]; then
    echo "Creating virtual environment..."
    "$INSTALL_PYTHON" -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade uv

if [[ "$CLEAR_PIP_CACHE" == "1" ]]; then
    _clear_pip_cache
fi

echo
echo "Detecting AMD GPU architecture..."
ARCH=""
if [[ ! -f detect_gpu_linux.py ]]; then
    echo "ERROR: detect_gpu_linux.py not found."
    exit 1
fi

if ! ARCH="$(python detect_gpu_linux.py 2>gpu_detect_debug.log)"; then
    echo "ERROR: GPU detection failed or unsupported AMD GPU."
    cat gpu_detect_debug.log 2>/dev/null || true
    exit 1
fi
ARCH="${ARCH//$'\r'/}"
ARCH="${ARCH//$'\n'/}"
if [[ -z "$ARCH" ]]; then
    echo "ERROR: Empty gfx code from detect_gpu_linux.py"
    exit 1
fi
echo "Detected GPU architecture: $ARCH"
echo "ROCm SDK pin: ${ROCM_SDK_PIN} (bitsandbytes libbitsandbytes_rocm714.so)"
echo

case "$ARCH" in
    gfx942)
        echo "Installing ROCm PyTorch for MI300/MI325 (gfx942)..."
        python -m pip install rocm[devel,libraries] \
            --index-url https://rocm.nightlies.amd.com/v2-staging/gfx942-dcgpu/
        rocm-sdk init || true
        python -m pip install --index-url https://rocm.nightlies.amd.com/v2-staging/gfx942-dcgpu/ \
            torch torchvision
        ;;
    gfx950)
        echo "Installing ROCm PyTorch for MI350/MI355 (gfx950)..."
        python -m pip install rocm[devel,libraries] \
            --index-url https://rocm.nightlies.amd.com/v2-staging/gfx950-dcgpu/
        rocm-sdk init || true
        python -m pip install --index-url https://rocm.nightlies.amd.com/v2-staging/gfx950-dcgpu/ \
            torch torchvision
        ;;
    *)
        if [[ "$ROCM_CHANNEL" == "stable" ]]; then
            echo "Installing ROCm PyTorch ${ROCM_SDK_PIN} (stable multi-arch from repo.amd.com) for ${ARCH}..."
            install_torch_stable || {
                echo "ERROR: Stable ROCm PyTorch install failed for ${ARCH}." >&2
                echo "       repo.amd.com only — no nightly fallback (set ROCM_CHANNEL=nightly for nightlies)." >&2
                echo "       Override: TORCH_PIN=… (must install amd-torch-device-${ARCH})." >&2
                exit 1
            }
        else
            echo "Installing ROCm PyTorch ${ROCM_SDK_PIN} (multi-arch nightly builds) for ${ARCH}..."
            if ! install_torch_nightly; then
                echo "Nightly ${ROCM_SDK_PIN} unavailable — trying stable repo.amd.com..."
                install_torch_stable || {
                    echo "ERROR: Could not install torch with ROCm ${ROCM_SDK_PIN}."
                    echo "       bitsandbytes requires libbitsandbytes_rocm714.so (BNB_ROCM_VERSION=714)."
                    exit 1
                }
            fi
        fi
        ;;
esac

echo
echo "Installing Fizgig dependencies..."
python -m uv pip install --link-mode copy --index-strategy unsafe-best-match \
    -r requirements-global.txt

echo
echo "Installing bitsandbytes (ROCm 7.14 — libbitsandbytes_rocm714.so)..."
python -m pip install -U -r requirements-rocm-linux.txt

echo
echo "Verifying bitsandbytes ROCm 7.14 library..."
python - <<'PY'
import sys
from pathlib import Path
import bitsandbytes

so = Path(bitsandbytes.__file__).resolve().parent / "libbitsandbytes_rocm714.so"
if not so.is_file():
    print(f"ERROR: missing {so}", file=sys.stderr)
    sys.exit(1)
print(f"OK  {so.name} ({so.stat().st_size:,} bytes)")
PY

echo
echo "Verifying ROCm / HIP..."
python - <<'PY'
import torch
print(f"PyTorch {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"HIP: {getattr(torch.version, 'hip', 'n/a')}")
print(f"ROCm: {getattr(torch.version, 'rocm', 'n/a')}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
PY
verify_torch_rocm_pin || exit 1
"$(_fizgig_rocm_python)" - <<'PY'
import torch
try:
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    _ = x @ x
    torch.cuda.synchronize()
    print("OK  GPU kernel smoke test passed")
except Exception as exc:
    import sys
    print(f"ERROR: GPU kernel smoke test failed: {exc}", file=sys.stderr)
    sys.exit(1)
PY

echo
echo "Checking VRAM monitor (amd-smi)..."
python - <<'PY'
import sys
sys.path.insert(0, "src")
from fizgig.utils.vram_monitor import _read_vram_amd_smi_cli
hit = _read_vram_amd_smi_cli()
if hit:
    used, total = hit
    print(f"OK  VRAM monitor: {used / (1024**3):.1f} / {total / (1024**3):.1f} GB in use")
else:
    print("WARN VRAM monitor: amd-smi unavailable or returned no data.")
    print("     Optional: sudo apt install amdrocm-amdsmi (see requirements-rocm-linux.txt)")
PY

echo
echo "Downloading InsightFace models (CPU, ~300 MB)..."
python - <<'PY'
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from insightface.app import FaceAnalysis
app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "genderage", "recognition"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1)
print("Models ready.")
PY

echo
echo "Writing ROCm launcher config (BNB_ROCM_VERSION, rocm_env.sh)..."
python write_rocm_env.py || true
chmod +x run_fizgig_rocm.sh

echo
echo "============================================================"
echo "  Installation complete!"
echo
echo "  Launch with: ./run_fizgig_rocm.sh"
echo "  (run_fizgig.sh is the upstream launcher — no ROCm env; do not use for AMD training)"
echo
echo "  NOTE: Linux AMD ROCm is highly experimental — crashes and GPU resets are common."
echo "============================================================"
