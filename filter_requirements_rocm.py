#!/usr/bin/env python3
"""Build a ROCm-safe requirements file from the shared requirements.txt.

Leaves requirements.txt untouched (CUDA path). Skips packages the AMD installers
install separately (torch / torchvision / bitsandbytes) and the CUDA
--extra-index-url line so a ROCm torch install is not replaced by cu128 wheels.

Usage:
  python filter_requirements_rocm.py [requirements.txt] [output.txt]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Installed by install_fizgig_rocm.bat / install_fizgig_rocm.sh instead.
SKIP_PACKAGES = frozenset({"torch", "torchvision", "bitsandbytes"})


def _package_name(line: str) -> str | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.startswith("-"):
        return None
    for sep in ("===", "==", ">=", "<=", "~=", "!=", ">", "<"):
        if sep in s:
            return s.split(sep, 1)[0].strip().lower().split("[")[0]
    return s.split()[0].lower().split("[")[0]


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "requirements.txt")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "requirements-rocm-shared.txt")
    if not src.is_file():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return 1

    kept: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("--extra-index-url") and "pytorch.org" in s:
            continue
        name = _package_name(line)
        if name in SKIP_PACKAGES:
            continue
        kept.append(line)

    out.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
