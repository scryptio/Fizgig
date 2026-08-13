#!/usr/bin/env python3
"""Launch lora_trainer_gui.py with ROCm debug logging enabled.

Usage (via run_fizgig_rocm.sh --debug, or directly):
  FIZGIG_ROCM_DEBUG=1 ./run_fizgig_rocm.sh
  python -u rocm_debug_launch.py

Logs are written under logs/rocm_debug_YYYYMMDD_HHMMSS.log unless FIZGIG_ROCM_LOG is set.
"""
from __future__ import annotations

import ast
import os
import sys
import types

import rocm_debug_hooks
from rocm_debug_hooks import _is_main_guard, patch_gui_namespace


def _run_gui_with_log_patch(target: str) -> None:
    """Execute lora_trainer_gui.py, patching the GUI logger before __main__ runs."""
    with open(target, encoding="utf-8") as handle:
        source = handle.read()

    tree = ast.parse(source, target)
    main_idx = next((i for i, node in enumerate(tree.body) if _is_main_guard(node)), None)

    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__file__": os.path.abspath(target),
        "__package__": None,
        "__cached__": None,
        "__doc__": None,
        "__spec__": None,
        "__builtins__": __builtins__,
    }
    sys.modules["__main__"] = types.ModuleType("__main__")
    sys.modules["__main__"].__dict__.update(namespace)

    if main_idx is None:
        code = compile(tree, target, "exec")
        exec(code, namespace)  # noqa: S102
        return

    pre_main = ast.Module(body=tree.body[:main_idx], type_ignores=getattr(tree, "type_ignores", []))
    post_main = ast.Module(body=tree.body[main_idx:], type_ignores=getattr(tree, "type_ignores", []))

    exec(compile(pre_main, target, "exec"), namespace)  # noqa: S102
    patched = patch_gui_namespace(namespace)
    if patched == 0:
        print("[AMD-ROCm debug] warning: GUI logger patch not applied", flush=True)
    exec(compile(post_main, target, "exec"), namespace)  # noqa: S102


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    target = os.path.join(root, "lora_trainer_gui.py")
    sys.argv[0] = target
    _run_gui_with_log_patch(target)
