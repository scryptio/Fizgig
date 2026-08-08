"""GPU VRAM reads for the status bar.

NVIDIA: pynvml (fast) then nvidia-smi.
AMD on Windows: typeperf ``\\GPU Local Adapter Memory(*)\\Local Usage``.
AMD on Linux: ``amd-smi`` CLI (ROCm Core / amdrocm-amdsmi), legacy ``rocm-smi``, else torch
    mem_get_info (allocator-only on HIP).

The PyPI ``amdsmi`` package is stale — AMD SMI now ships with ROCm Core SDK / the
``amdrocm-amdsmi`` system package. See:
https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import subprocess
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Windows performance counter — bytes of local VRAM in use per adapter.
_TYPEPERF_COUNTER = r"\GPU Local Adapter Memory(*)\Local Usage"
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _parse_typeperf_line(line: str) -> Optional[int]:
    """Return max adapter Local Usage (bytes) from one typeperf CSV row."""
    line = line.strip()
    if not line or line.startswith('"('):
        return None
    max_usage = 0.0
    for value in line.strip().split('","')[1:]:
        try:
            usage = float(value.strip('"').strip())
            if usage > max_usage:
                max_usage = usage
        except ValueError:
            pass
    return int(max_usage) if max_usage > 0 else None


def _total_vram_from_torch() -> Optional[int]:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    return None


def _coerce_bytes(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        m = re.match(r"^([\d.]+)\s*(B|KB|MB|GB|MiB|GiB|KIB|MIB|GIB)?$", value.strip(), re.I)
        if m:
            num = float(m.group(1))
            unit = (m.group(2) or "B").upper()
            scale = {"B": 1, "KB": 1024, "KIB": 1024, "MB": 1024 ** 2, "MIB": 1024 ** 2,
                     "GB": 1024 ** 3, "GIB": 1024 ** 3}.get(unit, 1)
            return int(num * scale)
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, dict):
        for key in (
            "vram_used", "vram_total", "used", "total",
            "VRAM Used Memory (B)", "VRAM Total Memory (B)",
            "VRAM Total Used Memory (B)", "VRAM_USED", "VRAM_TOTAL",
            "SIZE", "size",
        ):
            if key in value:
                hit = _coerce_bytes(value[key])
                if hit is not None:
                    return hit
        for v in value.values():
            hit = _coerce_bytes(v)
            if hit is not None:
                return hit
    return None


def _to_bytes(num: float, unit: Optional[str]) -> int:
    u = (unit or "MB").upper()
    scale = {"B": 1, "KB": 1024, "KIB": 1024, "MB": 1024 ** 2, "MIB": 1024 ** 2,
             "GB": 1024 ** 3, "GIB": 1024 ** 3}.get(u, 1024 ** 2)
    return int(num * scale)


def _linux_rocm_cli_env() -> dict[str, str]:
    """PATH/LD_LIBRARY_PATH so amd-smi from ROCm Core SDK is discoverable."""
    env = os.environ.copy()
    path_extra: list[str] = []
    for pattern in ("/opt/rocm/core-*/bin", "/opt/rocm/bin"):
        path_extra.extend(sorted(d for d in glob.glob(pattern) if os.path.isdir(d)))
    if path_extra:
        env["PATH"] = os.pathsep.join(path_extra + [env.get("PATH", "")])
    lib_extra = [p for p in ("/opt/rocm/lib", "/opt/rocm/lib64") if os.path.isdir(p)]
    if lib_extra:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_extra + [env.get("LD_LIBRARY_PATH", "")])
    return env


def _run_linux_cli(args: list[str], timeout: float = 4) -> Optional[str]:
    try:
        out = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            env=_linux_rocm_cli_env(),
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    except Exception:
        pass
    return None


def _read_vram_nvidia() -> Optional[tuple[int, int]]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        m = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(m.used), int(m.total)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
            creationflags=_CREATE_NO_WINDOW,
        )
        used, total = out.stdout.strip().splitlines()[0].split(",")
        return int(used.strip()) * 1024 * 1024, int(total.strip()) * 1024 * 1024
    except Exception:
        pass
    return None


_amd_smi_bin: Optional[str] = None


def _find_amd_smi() -> Optional[str]:
    global _amd_smi_bin
    if _amd_smi_bin is not None:
        return _amd_smi_bin or None
    env = _linux_rocm_cli_env()
    for name in ("amd-smi", "rocm-smi"):
        path = env.get("PATH", os.environ.get("PATH", ""))
        for directory in path.split(os.pathsep):
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                if name == "amd-smi":
                    _amd_smi_bin = candidate
                    return candidate
    for pattern in ("/opt/rocm/core-*/bin/amd-smi", "/opt/rocm/bin/amd-smi"):
        for candidate in sorted(glob.glob(pattern)):
            if os.access(candidate, os.X_OK):
                _amd_smi_bin = candidate
                return candidate
    _amd_smi_bin = ""
    return None


def _parse_vram_json(data) -> Optional[tuple[int, int]]:
    used_keys = (
        "VRAM Total Used Memory (B)", "VRAM Used Memory (B)", "VRAM_USED",
        "vram_used", "Used Memory (B)", "used", "mem_usage", "vram_used_mb",
    )
    total_keys = (
        "VRAM Total Memory (B)", "VRAM_TOTAL", "vram_total", "Total Memory (B)",
        "total", "SIZE", "size", "vram_total_mb",
    )

    def _walk(node):
        if isinstance(node, dict):
            used = total = None
            for k in used_keys:
                if k in node and node[k] is not None:
                    used = _coerce_bytes(node[k])
                    if used is not None:
                        break
            for k in total_keys:
                if k in node and node[k] is not None:
                    total = _coerce_bytes(node[k])
                    if total is not None:
                        break
            if used is not None and total is not None:
                return int(used), int(total)
            for v in node.values():
                hit = _walk(v)
                if hit:
                    return hit
        elif isinstance(node, list):
            for item in node:
                hit = _walk(item)
                if hit:
                    return hit
        return None

    return _walk(data)


def _parse_amd_smi_monitor_text(text: str) -> Optional[tuple[int, int]]:
    """Parse ``amd-smi monitor`` text — GPU 0 row, VRAM_USED / VRAM_TOTAL columns."""
    for line in text.splitlines():
        if line.startswith("GPU") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        if parts[0] != "0":
            continue
        # ... N/A 0 % 14 MB 96432 MB  -> last two numeric+unit pairs
        nums: list[tuple[float, Optional[str]]] = []
        i = 0
        while i < len(parts):
            if re.match(r"^[\d.]+$", parts[i]):
                unit = parts[i + 1] if i + 1 < len(parts) else None
                if unit and unit.upper() in {"MB", "GB", "MIB", "GIB", "B", "KB", "KIB"}:
                    nums.append((float(parts[i]), unit))
                    i += 2
                    continue
                nums.append((float(parts[i]), "MB"))
            i += 1
        if len(nums) >= 2:
            used = _to_bytes(nums[-2][0], nums[-2][1])
            total = _to_bytes(nums[-1][0], nums[-1][1])
            return used, total
    return None


def _read_vram_amd_smi_cli() -> Optional[tuple[int, int]]:
    """Linux AMD: amd-smi from ROCm Core SDK / amdrocm-amdsmi (replaces legacy rocm-smi)."""
    if _find_amd_smi() is None:
        return None
    bin_path = _amd_smi_bin
    assert bin_path

    for extra in (
        ["monitor", "--vram-usage", "--json"],
        ["monitor", "-v", "--json"],
        ["--json"],
        ["static", "--vram", "--json"],
    ):
        text = _run_linux_cli([bin_path, *extra])
        if not text:
            continue
        try:
            hit = _parse_vram_json(json.loads(text))
            if hit:
                return hit
        except json.JSONDecodeError:
            pass
        hit = _parse_amd_smi_monitor_text(text)
        if hit:
            return hit

    text = _run_linux_cli([bin_path, "monitor", "--vram-usage"])
    if text:
        hit = _parse_amd_smi_monitor_text(text)
        if hit:
            return hit
    return None


def _parse_rocm_smi_text(text: str) -> Optional[tuple[int, int]]:
    """Parse ``rocm-smi --showmeminfo vram`` plain-text output."""
    used = total = None
    for line in text.splitlines():
        if "Used Memory (B)" in line or "Used Memory (MiB)" in line:
            m = re.search(r":\s*(\d+)", line)
            if m:
                val = int(m.group(1))
                if "MiB" in line:
                    val *= 1024 * 1024
                used = val
        if "Total Memory (B)" in line or "Total Memory (MiB)" in line:
            m = re.search(r":\s*(\d+)", line)
            if m:
                val = int(m.group(1))
                if "MiB" in line:
                    val *= 1024 * 1024
                total = val
    if used is not None and total is not None:
        return used, total
    return None


def _read_vram_rocm_smi() -> Optional[tuple[int, int]]:
    """Linux AMD: legacy rocm-smi CLI (superseded by amd-smi in ROCm 7+)."""
    try:
        text = _run_linux_cli(["rocm-smi", "--showmeminfo", "vram", "--json"])
        if text:
            hit = _parse_vram_json(json.loads(text))
            if hit:
                return hit
    except Exception:
        pass
    try:
        text = _run_linux_cli(["rocm-smi", "--showmeminfo", "vram"])
        if text:
            hit = _parse_rocm_smi_text(text)
            if hit:
                return hit
    except Exception:
        pass
    return None


def _read_vram_torch_mem_info() -> Optional[tuple[int, int]]:
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(0)
            return int(total_b - free_b), int(total_b)
    except Exception:
        pass
    return None


class _TypeperfVramReader:
    """Background typeperf process — one sample per second, latest value cached."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_used: Optional[int] = None
        self._total: Optional[int] = None
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._failed = False

    def _reader_loop(self) -> None:
        proc = None
        try:
            proc = subprocess.Popen(
                ["typeperf", _TYPEPERF_COUNTER, "-si", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                creationflags=_CREATE_NO_WINDOW,
            )
            with self._lock:
                self._proc = proc
            assert proc.stdout is not None
            next(proc.stdout, None)
            next(proc.stdout, None)
            for line in proc.stdout:
                used = _parse_typeperf_line(line)
                if used is not None:
                    with self._lock:
                        self._latest_used = used
        except Exception as exc:
            logger.debug("typeperf VRAM reader stopped: %s", exc)
            with self._lock:
                self._failed = True
        finally:
            with self._lock:
                self._proc = None
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _ensure_started(self) -> None:
        with self._lock:
            if self._thread is not None or self._failed:
                return
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

    def _sample_once(self) -> Optional[int]:
        try:
            out = subprocess.run(
                ["typeperf", _TYPEPERF_COUNTER, "-sc", "1"],
                capture_output=True, text=True, timeout=10,
                creationflags=_CREATE_NO_WINDOW,
            )
            if out.returncode != 0:
                return None
            used = None
            for line in out.stdout.splitlines():
                hit = _parse_typeperf_line(line)
                if hit is not None:
                    used = hit
            return used
        except Exception:
            return None

    def read(self) -> Optional[tuple[int, int]]:
        if self._total is None:
            self._total = _total_vram_from_torch()
        self._ensure_started()
        with self._lock:
            used = self._latest_used
            failed = self._failed
        if used is None and not failed:
            used = self._sample_once()
            if used is not None:
                with self._lock:
                    self._latest_used = used
        if used is None or self._total is None:
            return None
        return used, self._total


def _is_rocm_backend() -> bool:
    if os.environ.get("FIZGIG_GPU_BACKEND") == "rocm":
        return True
    try:
        from .gpu_backend import is_rocm
        return is_rocm()
    except Exception:
        return False


def _use_windows_typeperf() -> bool:
    return os.name == "nt" and _is_rocm_backend()


def _use_linux_rocm() -> bool:
    return os.name != "nt" and _is_rocm_backend()


def _read_vram_linux_rocm() -> Optional[tuple[int, int]]:
    return _read_vram_amd_smi_cli() or _read_vram_rocm_smi() or _read_vram_torch_mem_info()


class VramMonitor:
    """Pick the best VRAM read path once, then reuse it."""

    def __init__(self, reader: Callable[[], Optional[tuple[int, int]]]) -> None:
        self._reader = reader

    @classmethod
    def create(cls) -> "VramMonitor":
        if _use_windows_typeperf():
            logger.debug("VRAM monitor: Windows typeperf (AMD ROCm)")
            return cls(_TypeperfVramReader().read)
        if _use_linux_rocm():
            logger.debug("VRAM monitor: Linux amd-smi / rocm-smi (AMD ROCm)")
            return cls(_read_vram_linux_rocm)
        logger.debug("VRAM monitor: NVIDIA pynvml / nvidia-smi / torch fallback")

        def _nvidia():
            return _read_vram_nvidia() or _read_vram_torch_mem_info()

        return cls(_nvidia)

    def read(self) -> Optional[tuple[int, int]]:
        try:
            return self._reader()
        except Exception as exc:
            logger.debug("VRAM read failed: %s", exc)
            return None


_monitor: Optional[VramMonitor] = None
_monitor_lock = threading.Lock()


def read_gpu_vram() -> Optional[tuple[int, int]]:
    """Return ``(used_bytes, total_bytes)`` for GPU 0, or ``None``."""
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            _monitor = VramMonitor.create()
    return _monitor.read()
