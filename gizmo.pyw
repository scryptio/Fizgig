"""Gizmo launcher — double-click to run without a console window.

The same shape as launch.pyw, and for the same reason: pythonw.exe hides the console, which
also hides FAILURES. If the app dies on import the window simply never appears — no error, no
clue. So every startup failure is reported through two channels that cannot themselves depend on
what broke: a native Windows message box via ctypes (deliberately not a Tkinter dialog, since
missing Tkinter is the classic cause) and gizmo_error.log beside this file.

Gizmo has no dependency Fizgig does not already have — Tkinter is stdlib, PIL and imageio-ffmpeg
are pinned in requirements.txt — so this needs no install of its own and nothing to re-run.
"""
import os
import runpy
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHONW = os.path.join(HERE, "venv", "Scripts", "pythonw.exe")
LOG = os.path.join(HERE, "gizmo_error.log")


def _report(title, message, detail=""):
    """The failure path. Must not use Tkinter, and must not raise."""
    logged = False
    try:
        with open(LOG, "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n{message}\n\n{detail}\n"
                    f"\npython: {sys.executable}\nversion: {sys.version}\n")
        logged = True
    except Exception:
        pass
    body = message + (f"\n\nDetails saved to:\n{LOG}" if logged else "")
    try:
        import ctypes
        # 0x10 = error icon, 0x10000 = bring the box to the foreground
        ctypes.windll.user32.MessageBoxW(0, body, title, 0x10010)
    except Exception:
        # Not Windows, or no usable GUI session. Guarded because stderr is None under pythonw.
        if getattr(sys, "stderr", None):
            sys.stderr.write(f"{title}\n{body}\n{detail}\n")
    sys.exit(1)


# If the venv exists and we're not already running from it, re-launch
if os.path.exists(VENV_PYTHONW) and os.path.normcase(sys.executable) != os.path.normcase(VENV_PYTHONW):
    try:
        subprocess.Popen([VENV_PYTHONW, os.path.abspath(__file__)])
    except Exception as exc:
        _report("Gizmo could not start",
                "The bundled Python failed to launch:\n"
                f"{VENV_PYTHONW}\n\n"
                "Re-run install_fizgig.bat to repair the environment.",
                f"{type(exc).__name__}: {exc}")
    sys.exit(0)

try:
    import tkinter  # noqa: F401
except Exception as exc:
    _report(
        "Gizmo — Python is missing Tkinter",
        "This Python was installed without Tkinter, which Gizmo needs for its interface.\n\n"
        "Reinstall Python from python.org and tick “tcl/tk and IDLE” during setup, then run "
        "install_fizgig.bat again.",
        f"{type(exc).__name__}: {exc}")

try:
    if os.path.isfile(LOG):
        os.remove(LOG)          # stale report from an earlier failed start
except Exception:
    pass

sys.path.insert(0, HERE)
try:
    runpy.run_path(os.path.join(HERE, "gizmo.py"), run_name="__main__")
except (SystemExit, KeyboardInterrupt):
    raise                       # deliberate exits are not errors
except BaseException as exc:
    _report("Gizmo could not start",
            f"Gizmo hit an error while starting:\n\n{type(exc).__name__}: {exc}\n\n"
            "If this keeps happening, please open a GitHub issue and attach the log below.",
            traceback.format_exc())
