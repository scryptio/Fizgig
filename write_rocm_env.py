#!/usr/bin/env python3
"""Probe the installed PyTorch ROCm version and write rocm_env.bat / rocm_env.sh.

bitsandbytes needs BNB_ROCM_VERSION to match the ROCm SDK bundled with the PyTorch wheel
(e.g. 714 for ROCm 7.14, 715 for 7.15). run_fizgig_rocm.bat and run_fizgig_rocm.sh source these.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
OUT_BAT = SCRIPT_DIR / "rocm_env.bat"
OUT_SH = SCRIPT_DIR / "rocm_env.sh"
DEFAULT_BNB_LINUX = "714"  # libbitsandbytes_rocm714.so - matches stable repo.amd.com torch
DEFAULT_BNB_WIN = "715"  # Windows 0xDELUXA wheel / pinned multi-arch torch (+rocm7.15)


def _bnb_lib_ext() -> str:
    return ".dll" if sys.platform == "win32" else ".so"


def _bitsandbytes_lib_dir() -> Path | None:
    try:
        import bitsandbytes
    except Exception:
        return None
    return Path(bitsandbytes.__file__).resolve().parent


def _bitsandbytes_rocm_lib(bnb_suffix: str) -> Path | None:
    lib_dir = _bitsandbytes_lib_dir()
    if lib_dir is None:
        return None
    lib = lib_dir / f"libbitsandbytes_rocm{bnb_suffix}{_bnb_lib_ext()}"
    return lib if lib.is_file() else None


def _bitsandbytes_rocm_libs_available() -> list[Path]:
    lib_dir = _bitsandbytes_lib_dir()
    if lib_dir is None:
        return []
    return sorted(lib_dir.glob(f"libbitsandbytes_rocm*{_bnb_lib_ext()}"))


def _windows_rocm_path_prefix(core: Path) -> str | None:
    """Directories where hipInfo.exe lives (bitsandbytes cuda_specs runs ``hipinfo``)."""
    parts: list[str] = []
    bin_dir = core / "bin"
    if bin_dir.is_dir():
        parts.append(str(bin_dir.resolve()))
    devel_bin = core.parent / "_rocm_sdk_devel" / "bin"
    if devel_bin.is_dir():
        parts.append(str(devel_bin.resolve()))
    venv_scripts = (SCRIPT_DIR / "venv" / "Scripts").resolve()
    if venv_scripts.is_dir():
        parts.append(str(venv_scripts))
    return ";".join(parts) if parts else None


def _linux_rocm_core_path() -> Path:
    for p in sorted((SCRIPT_DIR / "venv" / "lib").glob("python*/site-packages/_rocm_sdk_core")):
        if p.is_dir():
            return p.resolve()
    raise FileNotFoundError("_rocm_sdk_core not found under venv/lib/")


def bnb_rocm_version_from_torch() -> str:
    import torch

    rocm = getattr(torch.version, "rocm", None)
    if rocm:
        m = re.match(r"(\d+)\.(\d+)", str(rocm))
        if m:
            major = int(m.group(1))
            minor = int(m.group(2))
            # bitsandbytes wheels ship 64/70/71/714/72 — ignore ROCm 10.x style tags.
            if major >= 7 and major <= 9:
                return f"{major}{minor}"

    m = re.search(r"\+rocm(\d+)\.(\d+)", torch.__version__, re.I)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2))
        if major >= 7 and major <= 9:
            return f"{major}{minor}"

    hip = getattr(torch.version, "hip", None)
    if hip:
        m = re.match(r"(\d+)\.(\d+)", str(hip))
        if m:
            major = int(m.group(1))
            minor = int(m.group(2))
            if major >= 7 and major <= 9:
                return f"{major}{minor}"

    raise RuntimeError(
        f"could not parse ROCm 7.x version from torch {torch.__version__!r} "
        f"(rocm={rocm!r}, hip={hip!r})"
    )


def main() -> int:
    is_linux = sys.platform.startswith("linux")
    default_bnb = DEFAULT_BNB_LINUX if is_linux else DEFAULT_BNB_WIN
    lib_ext = _bnb_lib_ext()
    try:
        probed = bnb_rocm_version_from_torch()
        import torch
        src = f"torch {torch.__version__}"
        bnb = probed
        bnb_lib = _bitsandbytes_rocm_lib(bnb)
        if bnb_lib is None and is_linux and probed != DEFAULT_BNB_LINUX:
            fallback = _bitsandbytes_rocm_lib(DEFAULT_BNB_LINUX)
            if fallback is not None:
                print(
                    f"WARN: libbitsandbytes_rocm{bnb}{lib_ext} missing; "
                    f"using BNB_ROCM_VERSION={DEFAULT_BNB_LINUX} ({fallback.name})",
                    file=sys.stderr,
                )
                bnb = DEFAULT_BNB_LINUX
                bnb_lib = fallback
        if bnb_lib is None:
            available = _bitsandbytes_rocm_libs_available()
            print(
                f"WARN: no matching libbitsandbytes_rocm*{lib_ext} for ROCm {probed}; "
                f"using BNB_ROCM_VERSION={default_bnb}",
                file=sys.stderr,
            )
            if available:
                print(
                    "      Available:",
                    ", ".join(p.name for p in available),
                    file=sys.stderr,
                )
            bnb = default_bnb
            bnb_lib = _bitsandbytes_rocm_lib(bnb)
        elif is_linux and probed == DEFAULT_BNB_LINUX:
            src = f"{src} (Linux ROCm 7.14 / bnb {bnb})"
        else:
            src = f"{src} (bnb {bnb} / {bnb_lib.name})"
    except Exception as exc:
        print(f"WARN: {exc}; using default BNB_ROCM_VERSION={default_bnb}")
        bnb = default_bnb
        src = "default (install probe failed)"

    rocm_core_win = SCRIPT_DIR / "venv" / "Lib" / "site-packages" / "_rocm_sdk_core"
    try:
        rocm_core_linux = _linux_rocm_core_path()
    except FileNotFoundError:
        rocm_core_linux = None

    if rocm_core_win.is_dir():
        core = rocm_core_win.resolve()
        bat_lines = [
            "@echo off",
            "REM Auto-generated by write_rocm_env.py - re-run installer to refresh.",
            f"REM Detected from: {src}",
            f'set "BNB_ROCM_VERSION={bnb}"',
            f'set "ROCM_PATH={core}"',
            f'set "HIP_PATH={core}"',
        ]
        path_prefix = _windows_rocm_path_prefix(core)
        if path_prefix:
            bat_lines.append(f'set "PATH={path_prefix};%PATH%"')
        bat_lines.append("")
        OUT_BAT.write_text("\r\n".join(bat_lines), encoding="utf-8")
        print(f"Wrote {OUT_BAT.name}: BNB_ROCM_VERSION={bnb}  ({src})")

    if rocm_core_linux is not None:
        sh_lines = [
            "# Auto-generated by write_rocm_env.py - re-run installer to refresh.",
            f"# Detected from: {src}",
            f'export BNB_ROCM_VERSION="{bnb}"',
            f'export ROCM_PATH="{rocm_core_linux}"',
            f'export HIP_PATH="{rocm_core_linux}"',
            "",
        ]
        OUT_SH.write_text("\n".join(sh_lines), encoding="utf-8")
        print(f"Wrote {OUT_SH.name}: BNB_ROCM_VERSION={bnb}  ({src})")

    if not rocm_core_win.is_dir() and rocm_core_linux is None:
        print("WARN: _rocm_sdk_core not found - no rocm_env files written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
