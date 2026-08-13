"""Temporary ROCm debug hooks — tee stdout/stderr and subprocess output to a log file.

Loaded by rocm_debug_launch.py before lora_trainer_gui.py starts. Do not import from
application code; this exists only for GPU crash diagnosis on AMD/ROCm.
"""
from __future__ import annotations

import ast
import atexit
import datetime as _dt
import faulthandler
import os
import signal
import subprocess
import sys
import threading
import traceback
from typing import Any, IO, Optional, TextIO

_LOG_LOCK = threading.Lock()
_LOG_FILE: Optional[TextIO] = None
_LOG_PATH: Optional[str] = None
_ORIG_STDOUT = sys.stdout
_ORIG_STDERR = sys.stderr
_ORIG_POPEN = subprocess.Popen
_FD_TEE_INSTALLED = False
_INSTALLED = False


def _log_dir() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.environ.get("FIZGIG_ROCM_LOG_DIR") or os.path.join(root, "logs")
    os.makedirs(path, exist_ok=True)
    return path


def _open_log_file() -> TextIO:
    global _LOG_FILE, _LOG_PATH
    if _LOG_FILE is not None:
        return _LOG_FILE

    explicit = os.environ.get("FIZGIG_ROCM_LOG", "").strip()
    if explicit:
        _LOG_PATH = os.path.abspath(explicit)
        parent = os.path.dirname(_LOG_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        _LOG_PATH = os.path.join(_log_dir(), f"rocm_debug_{stamp}.log")

    _LOG_FILE = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)
    return _LOG_FILE


def _write_log(text: str) -> None:
    if not text:
        return
    with _LOG_LOCK:
        log = _open_log_file()
        log.write(text)
        if not text.endswith("\n"):
            log.write("\n")
        log.flush()


class _TeeReader:
    """Wrap a subprocess pipe so GUI readers still work, but lines are logged."""

    def __init__(self, stream: IO[str], label: str) -> None:
        self._stream = stream
        self._label = label

    def readline(self) -> str:
        line = self._stream.readline()
        if line:
            _write_log(f"[subprocess:{self._label}] {line.rstrip()}\n")
        return line

    def read(self, size: int = -1) -> str:
        data = self._stream.read(size)
        if data:
            for line in data.splitlines(keepends=True):
                _write_log(f"[subprocess:{self._label}] {line.rstrip()}\n")
        return data

    def close(self) -> None:
        self._stream.close()

    def __iter__(self):
        return self

    def __next__(self) -> str:
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _log_startup_banner() -> None:
    _write_log("=" * 72 + "\n")
    _write_log(f"Fizgig ROCm debug session started {_dt.datetime.now().isoformat(timespec='seconds')}\n")
    _write_log(f"log file: {_LOG_PATH}\n")
    _write_log(f"python: {sys.executable}\n")
    _write_log(f"argv: {sys.argv}\n")
    _write_log(f"cwd: {os.getcwd()}\n")

    env_keys = sorted(
        k for k in os.environ
        if any(
            token in k.upper()
            for token in (
                "ROCM", "HIP", "HSA", "MIOPEN", "PYTORCH", "TORCH", "BNB",
                "FIZGIG", "AMD", "FLASH", "CUDA", "GPU", "ALLOC",
            )
        )
    )
    _write_log("--- relevant environment ---\n")
    for key in env_keys:
        _write_log(f"{key}={os.environ.get(key, '')}\n")

    try:
        import torch

        _write_log("--- torch ---\n")
        _write_log(f"torch.__version__={torch.__version__}\n")
        if torch.cuda.is_available():
            _write_log(f"torch.cuda.device_count()={torch.cuda.device_count()}\n")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                _write_log(f"  device {i}: {props.name}  arch={getattr(props, 'gcnArchName', '?')}\n")
    except Exception as exc:
        _write_log(f"torch import probe failed: {type(exc).__name__}: {exc}\n")

    _write_log("=" * 72 + "\n")


def _log_shutdown(reason: str) -> None:
    _write_log(f"\n--- session ended ({reason}) {_dt.datetime.now().isoformat(timespec='seconds')} ---\n")
    if _LOG_FILE is not None:
        try:
            _LOG_FILE.flush()
        except Exception:
            pass


def _install_excepthook() -> None:
    def _hook(exc_type, exc, tb):
        _write_log("--- unhandled exception ---\n")
        _write_log("".join(traceback.format_exception(exc_type, exc, tb)))
        _orig_excepthook(exc_type, exc, tb)

    _orig_excepthook = sys.excepthook
    sys.excepthook = _hook


def _install_signal_logging() -> None:
    def _handler(signum, frame):
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = str(signum)
        _write_log(f"\n--- signal received: {name} ---\n")
        try:
            faulthandler.dump_traceback(file=_open_log_file(), all_threads=True)
        except Exception:
            pass

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


def _install_fd_tee() -> None:
    """Duplicate OS stdout/stderr so C libraries (HIP/MIOpen) are logged too."""
    global _FD_TEE_INSTALLED
    if _FD_TEE_INSTALLED:
        return
    if not hasattr(os, "pipe") or not hasattr(os, "dup2"):
        return

    try:
        orig_out = os.dup(1)
        orig_err = os.dup(2)
    except OSError:
        return

    r_out, w_out = os.pipe()
    r_err, w_err = os.pipe()

    os.dup2(w_out, 1)
    os.dup2(w_err, 2)
    os.close(w_out)
    os.close(w_err)

    sys.stdout = open(1, "w", encoding="utf-8", buffering=1, closefd=False)  # noqa: SIM115
    sys.stderr = open(2, "w", encoding="utf-8", buffering=1, closefd=False)  # noqa: SIM115

    def _relay(read_fd: int, orig_fd: int, label: str) -> None:
        read_file = open(read_fd, "rb", buffering=0)  # noqa: SIM115
        term = os.fdopen(orig_fd, "wb", buffering=0)
        while True:
            chunk = read_file.read(8192)
            if not chunk:
                break
            term.write(chunk)
            term.flush()
            text = chunk.decode("utf-8", errors="replace")
            if text:
                _write_log(f"[{label}] {text}")

    threading.Thread(target=_relay, args=(r_out, orig_out, "stdout"), daemon=True).start()
    threading.Thread(target=_relay, args=(r_err, orig_err, "stderr"), daemon=True).start()
    _FD_TEE_INSTALLED = True


class _TeePopen(_ORIG_POPEN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cmd = args[0] if args else kwargs.get("args")
        _write_log(f"[subprocess:start] {cmd}\n")
        if self.stdout is not None and hasattr(self.stdout, "readline"):
            self.stdout = _TeeReader(self.stdout, "stdout")  # type: ignore[assignment]
        if self.stderr is not None and hasattr(self.stderr, "readline"):
            self.stderr = _TeeReader(self.stderr, "stderr")  # type: ignore[assignment]


def patch_gui_class(cls: type) -> bool:
    """Mirror the in-app console buffer to the debug log file."""
    if getattr(cls, "_rocm_debug_patched", False):
        return False

    orig_append = cls._append_global_log

    def _append_global_log(self, text):
        if text:
            _write_log(text if text.endswith("\n") else f"{text}\n")
        return orig_append(self, text)

    cls._append_global_log = _append_global_log  # type: ignore[method-assign]
    cls._rocm_debug_patched = True  # type: ignore[attr-defined]
    _write_log(f"[debug] patched GUI logger on {cls.__name__}\n")
    return True


def patch_gui_namespace(namespace: dict[str, Any]) -> int:
    patched = 0
    for value in namespace.values():
        if isinstance(value, type) and hasattr(value, "_append_global_log"):
            if patch_gui_class(value):
                patched += 1
    return patched


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    return any(isinstance(comp, ast.Constant) and comp.value == "__main__" for comp in test.comparators)


def install() -> str:
    """Install tee hooks; returns the log file path."""
    global _INSTALLED
    if _INSTALLED:
        return _LOG_PATH or ""

    log = _open_log_file()
    faulthandler.enable(file=log, all_threads=True)

    _install_fd_tee()
    subprocess.Popen = _TeePopen  # type: ignore[misc,assignment]

    _install_excepthook()
    _install_signal_logging()
    atexit.register(lambda: _log_shutdown("normal exit"))
    _log_startup_banner()

    _INSTALLED = True
    print(f"[AMD-ROCm debug] logging to {_LOG_PATH}", flush=True)
    return _LOG_PATH or ""


_LOG_PATH = install()
