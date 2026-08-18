"""Device utilities for memory management and synchronization."""

import gc
import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


def fp8_scaled_mm_supported(device: Optional[Union[str, torch.device]] = None) -> bool:
    """True if the GPU has fp8 tensor cores usable by torch._scaled_mm.

    Requires compute capability >= 8.9 (Ada / Hopper / Blackwell). Older cards
    (Ampere sm_86 like the 3090, Turing, etc.) lack fp8 silicon and must fall
    back to the dequantize-to-bf16 path — for them this returns False and the
    fast path is never entered, so training/inference behaves exactly as today.
    """
    if not torch.cuda.is_available():
        return False
    if device is not None:
        dev = torch.device(device) if isinstance(device, str) else device
        index = dev.index if dev.type == "cuda" else None
    else:
        index = None
    try:
        major, minor = torch.cuda.get_device_capability(index)
    except Exception:
        return False
    return (major, minor) >= (8, 9)


def plannable_free_vram(device: Optional[Union[str, torch.device]] = None) -> float:
    """Free VRAM in GB for PLANNING decisions — honouring the small-card simulator.

    Set FIZGIG_SIM_VRAM_GB=16 and every planner behaves as though the machine had a 16 GB
    card: reported free becomes (simulated total − whatever Windows/desktop currently eat),
    the same view that card's real owner gets. A separate VRAM-hog process cannot do this
    job — WDDM virtualizes memory per process, so mem_get_info in Fizgig's processes never
    sees another process's ballast (the issue-#71 overcommit behaviour, met from the other
    side). Pair with apply_sim_vram_cap() so exceeding the budget genuinely OOMs too.
    """
    import os
    idx = None
    if device is not None:
        idx = torch.device(device).index
    free_b, total_b = torch.cuda.mem_get_info(idx if idx is not None else 0)
    free = free_b / 1e9
    sim = os.environ.get("FIZGIG_SIM_VRAM_GB", "").strip()
    if sim:
        try:
            reported_total = float(sim) * 0.995e9 / 1e9   # a "16 GB" card reports ~15.9
            deficit = (total_b - free_b) / 1e9            # the Windows/desktop tax
            free = min(free, max(0.0, reported_total - deficit))
        except ValueError:
            pass
    return free


def apply_sim_vram_cap(device: Optional[Union[str, torch.device]] = None):
    """The enforcement half of the simulator: cap this process's torch allocator at the
    simulated card size, so an allocation a real small card could not make OOMs here too
    instead of quietly spilling into the 5090's headroom. No-op without the env var."""
    import os
    sim = os.environ.get("FIZGIG_SIM_VRAM_GB", "").strip()
    if not sim or not torch.cuda.is_available():
        return
    try:
        idx = torch.device(device).index if device is not None else 0
        total = torch.cuda.mem_get_info(idx or 0)[1] / 1e9
        frac = min(1.0, (float(sim) * 0.995) / total)
        torch.cuda.set_per_process_memory_fraction(frac, idx or 0)
        logger.warning(f"[sim] FIZGIG_SIM_VRAM_GB={sim}: allocator capped at "
                       f"{frac * total:.1f} GB — this process behaves like a {sim} GB card.")
    except Exception as exc:
        logger.warning(f"[sim] could not cap the allocator: {exc}")


def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
    """Free cached memory on the specified device."""
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def gpu_svd(W: torch.Tensor) -> tuple:
    """SVD on GPU if available, CPU fallback. Returns (U, S, Vt) on CPU.

    A CUDA failure (usually OOM when the GPU is busy) falls back to CPU — but it's logged
    at WARNING so a slow, CPU-bound run is diagnosable instead of a silent mystery."""
    if torch.cuda.is_available():
        try:
            W_gpu = W.cuda()
            U, S, Vt = torch.linalg.svd(W_gpu, full_matrices=False)
            return U.cpu(), S.cpu(), Vt.cpu()
        except Exception as e:
            logger.warning("gpu_svd: CUDA SVD failed (%s: %s) for shape %s — falling back to CPU "
                           "(much slower). Free GPU memory to keep SVD on the GPU.",
                           type(e).__name__, e, tuple(W.shape))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return torch.linalg.svd(W, full_matrices=False)


def gpu_kron(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Kronecker product on GPU if available, CPU fallback. Returns result on CPU.

    Logs at WARNING on CUDA failure so a CPU fallback (usually OOM) is visible, not silent."""
    if torch.cuda.is_available():
        try:
            result = torch.kron(w1.cuda(), w2.cuda()).cpu()
            return result
        except Exception as e:
            logger.warning("gpu_kron: CUDA kron failed (%s: %s) for shapes %s x %s — falling back "
                           "to CPU. Free GPU memory to keep this on the GPU.",
                           type(e).__name__, e, tuple(w1.shape), tuple(w2.shape))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return torch.kron(w1, w2)


def synchronize_device(device: Optional[Union[str, torch.device]]):
    """Block until all pending operations on the device are complete."""
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
