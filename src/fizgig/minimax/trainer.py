"""MiniMax H3 — image-only training core: flow-matching loss + timestep sampling.

The heart of the trainer, isolated so it's headless-testable with the tiny model (no GPU,
no 66 GB base, no 32 B text encoder). The full LoRA/rotating-FT wiring, caching and GUI come
later; this pins the maths of one training step.

Flow / sign convention (matched to ComfyUI's comfy/ldm/minimax/model.py):
  x0 = clean latent, noise ~ N(0,1), sigma in (0,1) the noise level.
  noised = (1 - sigma)*x0 + sigma*noise            (sigma 0 = clean, 1 = pure noise)
  t = 1 - sigma                                     the "cleanness" fed to the time embedder
  the DiT's raw video_out predicts (x0 - noise)     (the reference NEGATES it to get the
                                                     sampler's velocity noise - x0)
So the training target for the model's output is `x0 - noise`.
"""

import argparse
import contextlib
import gc
import logging
import math
import os
import random
import re
import sys
import time
from multiprocessing import Value

import torch
import torch.nn.functional as F

from fizgig.training.metadata import ARCHITECTURE_MINIMAX

logger = logging.getLogger(__name__)

VIDEO_SIGMA_SHIFT_TRAIN = 12.0     # H3's video shift — also the reference TRAINING density

# Identity-first phase 1 trains at this fraction of the Learning Rate box (Peter, 11 Aug). Phase
# 1 places the identity on a near-zero adapter, where a full-size Adam stride does the most
# damage and the least good; phase 2 then gets the full rate from a sensible starting point.
_P1_LR_SCALE = 1.0 / 3.0

# LoRA targets the transformer blocks' ATTENTION + MLP Linears (+ the 2-block text refiner).
# The fp32 patch/head IO layers are left alone (wrapping them clashes fp32-base vs bf16-adapter).
#
# `adaln_proj` is per-checkpoint (matching the reference trainer on the pruned build):
#   * FULL bf16 model ([96768, 2688]): EXCLUDED — the up-matrices are 96768-out (6x qkv),
#     soaked up the largest share of LoRA capacity, and ComfyUI's pruned inference builds
#     drop every adaln key anyway (~50% likeness until excluded, real run).
#   * PRUNED model ([96768, 8]): INCLUDED — deploy-consistent, and what ai-toolkit trains.
#     It carries ~45% of all weight movement in a matched reference epoch, and it is the
#     timestep-conditioned modulation, so starving it reads from outside as "the mid/low-noise
#     range never gets trained". Train it at the REQUESTED rank: capping to min(in,out)=8 cost
#     73% of its learning (see the no-cap note in networks/lora.py). An epoch-1 melt was once
#     pinned on these adapters (tests/diag_epoch1_ab.py) but the distortion predated adaln and
#     persisted without it — the real culprit was the training density (see sample_sigmas).
DEFAULT_INCLUDE_PATTERNS = [r"blocks\.\d+\.attn\..*", r"blocks\.\d+\.mlp\..*",
                            r"token_refiner\.blocks\..*"]
# NOTE: the per-block AdaLNs only — NOT `final_layer.adaln_proj`. The reference trains 258
# modules and we were training 259; the extra one was added here by symmetry, not by matching
# them. It also happened to carry our single highest per-element drift after a matched epoch
# (0.0133 vs their 0.0068 max), so it was contributing noise rather than capability.
PRUNED_INCLUDE_PATTERNS = DEFAULT_INCLUDE_PATTERNS + [r"blocks\.\d+\.adaln_proj\..*"]


def clip_fallback_frames(frames: int) -> int:
    """Next shorter clip length to retry with after a clip preview fails (in practice, OOM).

    Halves the request and snaps down onto the model's 17n+5 grid, so a 141-frame OOM retries
    at 56, then 22, and only then gives up on clips: 141 -> 56 -> 22 -> 1.

    Stepping down rather than collapsing straight to a still matters because a still is the
    MOST out-of-distribution render H3 has — ComfyUI cannot even construct one (its video
    latent floor is 2 frames) and the trained band is ~124-362. Dropping a clip run to stills
    on one OOM quietly replaces the previews being judged with the least trustworthy kind,
    for the rest of the run. A shorter clip is still a clip.
    """
    half = int(frames) // 2
    if half < 22:                      # below the first real grid point above a keyframe pair
        return 1
    return half - (half - 5) % 17      # largest 17n+5 value <= half


def parse_block_spec(spec, num_blocks: int = None):
    """"3-12, 14-15, 22,27,31-33" -> [3,4,...,12,14,15,22,27,31,32,33].

    Ranges and singles, comma-separated, whitespace anywhere. Returns sorted unique indices.
    Raises ValueError on anything it cannot read — a typo here must stop the run, not silently
    train a different set of blocks than the one being tested.

    num_blocks, when given, bounds-checks: an out-of-range index would otherwise just match
    nothing and quietly shrink the experiment.
    """
    text = str(spec if spec is not None else "").strip()
    if not text:
        raise ValueError("no blocks given")
    out = set()
    for part in text.split(","):
        chunk = part.strip()
        if not chunk:
            continue                       # tolerate a trailing or doubled comma
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", chunk)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                raise ValueError(f"range runs backwards: {chunk!r}")
            out.update(range(lo, hi + 1))
        elif re.fullmatch(r"\d+", chunk):
            out.add(int(chunk))
        else:
            raise ValueError(f"cannot read {chunk!r} — use numbers and ranges, "
                             f"e.g. '3-12, 14-15, 22, 31-33'")
    if not out:
        raise ValueError("no blocks given")
    if num_blocks is not None:
        bad = sorted(i for i in out if i >= num_blocks)
        if bad:
            raise ValueError(f"block(s) {bad} do not exist — this model has {num_blocks} "
                             f"(0-{num_blocks - 1})")
    return sorted(out)


def format_block_spec(indices):
    """[3,4,5,7] -> "3-5,7" — the canonical form recorded in metadata and logged."""
    if not indices:
        return ""
    runs, start, prev = [], indices[0], indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def restrict_patterns_to_blocks(patterns, block_spec, num_blocks: int = None):
    """Narrow `blocks.N.*` patterns to a block selection. Non-block patterns pass through.

    H3 is 50 IDENTICAL blocks with no published map of what each one does, so training a subset is
    an experiment, not a recipe — this exists to make that experiment cheap to run. The token
    refiner is deliberately never narrowed: it is text-side (where a trigger token gets shaped),
    it is 8 of 258 modules, and holding it constant keeps two selections comparable to each other
    rather than confounding the block question with a conditioning change.

    Applied ON TOP of the per-checkpoint pattern list rather than replacing it, so the pruned vs
    bf16 AdaLN decision stays in exactly one place.
    """
    idx = parse_block_spec(block_spec, num_blocks)
    alt = "|".join(str(i) for i in idx)
    out = []
    for p in patterns:
        if p.startswith(r"blocks\.\d+"):
            out.append(p.replace(r"blocks\.\d+", rf"blocks\.(?:{alt})", 1))
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# VRAM planner — resolves "auto" block swap + gradient checkpointing from the card's actual
# free VRAM and the run's real token load (bucket megapixels x batch). Simpler than Krea 2's:
# one quant mode (NF4), batch is 1, no preview co-residency.
# ---------------------------------------------------------------------------
# Measured anchors (5090, real 33B, rank 16, ~0.2 MP batch 1 — GPU validation pass, 4 Aug):
#   no swap, no ckpt : resident 17.6, step peak 22.7  (overhead ~5.1)
#   no swap, ckpt    : resident 17.5, step peak 18.3  (overhead 0.9 — and only ~+0.1 s/step)
#   swap 16 + ckpt   : resident 11.9 (0.34 GB/block), steady 12.8, step peak 19.3 — the swap
#                      path carries a ~7.4 GB backward transient (checkpoint recompute segments
#                      held by the engine), which the planner must budget on top of residency.
#
# Re-measured 6 Aug on the SHIPPED default (int8 base, LoKR factor 8 + adamw, AdaLN off), because
# those anchors were taken with a rank-16 LoRA on adamw8bit — an adapter of ~0.4 GB against the
# ~3.1 GB the defaults now carry, so the planner was budgeting for a run nobody does:
#   resident         : base 21.07 + LoKR weights 0.63 + fp32 Adam state 2.50 = 24.20 GB
#   0.23 MP  no ckpt : 29.18      |  ckpt: 24.39
#   0.50 MP  no ckpt : OOM (>31)  |  ckpt: 24.47
#   0.98 MP  no ckpt : OOM        |  ckpt: 24.56
# Two things fall out. Un-checkpointed really does scale hard (0.5 MP OOMs a 32 GB card, so
# forcing ckpt on there is correct), and CHECKPOINTED IS ALMOST FLAT — 1 MP costs 0.17 GB more
# than 0.23 MP, not four times as much. Hence _ACT_GB_CKPT below.
_RESIDENT_GB = 17.5          # full bf16 model, NF4 resident (measured 17.3-17.6)
# The PRUNED checkpoint drops the full-width AdaLN (~40% of the model's weight mass) for a curve
# table, so the same NF4 pass lands far smaller: ~20.1 B params quantized -> ~10.1 GB, plus the
# unquantized remainder. Estimated from the file's own tensor census, not yet GPU-measured, so
# it carries margin.
# MEASURED 6 Aug (was 11.0, estimated from the file's tensor census): the pruned checkpoint
# decoded and re-quantized to NF4 sits at 10.46 GB resident, and a checkpointed step peaks at
# 13.46 / 13.56 / 13.63 GB at 0.23 / 0.50 / 0.98 MP — flat in megapixels, exactly like int8.
# Un-checkpointed it is 18.27 / 23.52 / OOM. Now that Auto can CHOOSE this mode, the number it
# chooses against had to stop being a guess.
_RESIDENT_PRUNED_GB = 10.5
# int8 base (base_quant=int8, the reference's own storage): the 200 block linears stay 1 byte
# per param instead of NF4's 0.5, and the refiner/AdaLN load dense — ~19.3 + ~1.5 GB.
_RESIDENT_INT8_GB = 21.0
# int8 dequantizes a bf16 weight per matmul (fc1 is 28672x5376 = 308 MB). A few are live at
# once, but they are NOT retained for backward — _Int8RotLinearFn recomputes the weight in its
# own backward, so the cost is a handful of transients rather than one per layer. (Before that
# custom backward, autograd saved every one and a 0.25 MP run OOM'd the moment the planner
# turned checkpointing off: measured 0.45 GB of retained weight over 12 test linears against
# 0.12 GB now, and the real DiT has 200.)
_INT8_TRANSIENT_GB = 1.0
_PER_BLOCK_GB = 0.34         # one parked block's GPU share (measured: (17.5-11.9)/16)
_ACT_GB_NOCKPT = 5.5         # step overhead at 0.25 MP batch 1, no checkpointing (measured 4.98)
# Checkpointed memory is very nearly FLAT in megapixels — that is the whole point of recompute,
# and the old 2.0 (which then got multiplied by the MP scale) modelled it as growing four times
# faster than it does. Measured on the shipped default (int8 base, LoKR 8 + adamw, 6 Aug 2026),
# peak above the resident 24.20 GB:
#     0.23 MP  0.19 GB        0.50 MP  0.27 GB        0.98 MP  0.36 GB
# i.e. ~0.15 + 0.2 x scale. 0.5 keeps a wide margin at every size and still leaves the planner
# free to say "no swap" where the card genuinely fits — the old value invented 25 blocks of swap
# for a 1 MP run that actually peaks at 24.6 GB, costing ~4x the step time for nothing.
_ACT_GB_CKPT = 0.5           # step overhead at 0.25 MP batch 1, checkpointed (measured 0.19)
_SWAP_TRANSIENT_GB = 7.5     # extra backward-time peak whenever swap is active (measured 7.4 @ n=16)
_RESERVE_GB = 1.5            # display / allocator / fragmentation headroom
# Skipping checkpointing has to EARN it. Measured on H3, recompute costs ~0.1 s/step and saves
# ~5 GB — so choosing "no checkpointing" on a thin margin trades five gigabytes of headroom for
# a tenth of a second. Peter's 6 Aug run picked it with 0.37 GB of predicted margin (needed
# 32.13 of 32.5 GB free) and then ran at 4-6 s/step instead of ~1: on Windows the driver spills
# to system RAM rather than OOMing, so an over-tight plan does not fail, it just crawls, with
# nothing in the log to say why. The un-checkpointed peak is also the one that scales with
# megapixels, so a plan that barely fits at one bucket size will not fit at the next.
_NOCKPT_MARGIN_GB = 3.0      # extra headroom demanded before skipping recompute


def adapter_param_count(dit_path: str, include_patterns, network_type: str = "lora",
                        network_dim: int = 16, lokr_factor: int = 8,
                        train_blocks: str = None) -> int:
    """Trainable parameter count, read from the checkpoint HEADER — no model, no GPU.

    The VRAM plan runs before the DiT is built, so the shapes come from the safetensors header
    (which is just JSON at the front of the file). That keeps this exact rather than an
    architecture guess: it sees the real targeted Linears for whichever checkpoint is loaded,
    respects include_patterns and the Blocks to Train restriction, and works the same on the
    pruned and full builds.
    """
    import json
    import re as _re
    import struct
    try:
        with open(dit_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
    except Exception:
        return 0

    pats = list(include_patterns or [])
    if train_blocks:
        n_blocks = len({int(m.group(1)) for k in hdr
                        for m in [_re.match(r"blocks\.(\d+)\.", k)] if m} or {0})
        pats = restrict_patterns_to_blocks(pats, train_blocks, n_blocks)
    if not pats:
        return 0
    rx = [_re.compile(p) for p in pats]

    total = 0
    for key, ent in hdr.items():
        if key == "__metadata__" or not key.endswith(".weight"):
            continue
        shape = ent.get("shape") or []
        if len(shape) != 2:                     # Linears only, as create_modules wraps
            continue
        name = key[:-len(".weight")]
        if not any(r.search(name) for r in rx):
            continue
        out_dim, in_dim = int(shape[0]), int(shape[1])
        if str(network_type).lower() == "lokr":
            from fizgig.networks.lora import factorization   # local: avoids a circular import
            a, _c = factorization(out_dim, int(lokr_factor))
            b, _d = factorization(in_dim, int(lokr_factor))
            total += a * b + _c * _d            # w1 (a,b) + w2 (c,d)
        else:
            total += int(network_dim) * (in_dim + out_dim)
    return total


def adapter_vram_gb(params: int, optimizer_type: str = "adamw8bit") -> float:
    """GB the adapter holds for the WHOLE run: bf16 weights + optimizer state.

    Not a rounding error at these sizes. LoKR factor 8 on H3 trains ~313 M parameters against a
    rank-16 LoRA's ~77 M, and the state dtype widens the gap again: fp32 Adam keeps two 4-byte
    moments per parameter where the 8-bit optimizers keep two 1-byte ones. LoKR + adamw is
    ~3.1 GB against ~0.4 GB for the rank-16 + adamw8bit configuration the original anchors were
    measured on — which is why planning without this term was planning for a run nobody does.

    Gradients are deliberately NOT counted here. They are transient, and fused AdamW frees them
    per parameter as it steps, so they never all coexist: measured, a checkpointed step peaks
    only 0.19 GB above this figure even though the gradients would be 0.63 GB if they were all
    live at once. They belong in the activation term's margin, not in the resident one.

    Verified against a real step (6 Aug 2026): base 21.07 + weights 0.63 + fp32 state 2.50 =
    24.20 GB resident, exactly what this returns for 313.1 M parameters on adamw.
    """
    key = (optimizer_type or "adamw8bit").lower()
    n_states = 1 if "lion" in key else 2        # Lion keeps momentum only
    state_bytes = (1 if "8bit" in key else 4) * n_states
    return params * (2 + state_bytes) / 1e9     # bf16 weight + optimizer state


def plan_base_quant(free_gb: float, pruned: bool, mp: float = 0.25, adapter_gb: float = 0.0):
    """Pick the base quantisation AND the swap plan together -> (mode, blocks_to_swap, ckpt, why).

    Choosing a swap count from VRAM alone, with the quantisation already fixed, produces the
    worst available outcome on mid-range cards: the int8 base is ~21 GB, so a 24 GB card cannot
    hold it and the planner parks 38 of 50 blocks on CPU — every one of them crossing PCIe every
    step, for roughly 4x the step time. The same file loaded 4-bit is ~11 GB and needs no swap at
    all. Krea 2 hit this exact failure and fixed it the same way (see _auto_krea2_strategy):
    quantisation and swap are one decision.

    Order of preference:
      1. int8, no swap  — the most accurate base (~0.17% error against the reference's own
                          storage) with no PCIe cost. Always preferred when it fits.
      2. 4-bit, no swap — trades base accuracy (~9.5% error) for keeping every block resident.
      3. 4-bit + swap   — 11 GB resident always parks fewer blocks than 21 GB would.

    The trade in step 2 is real and worth stating: a LoRA fitted on a 9.5%-perturbed base spends
    capacity correcting error that will not exist at inference, and it compounds with depth. It
    is chosen only when the alternative is most of the model crossing PCIe on every step.

    Only applies to a pruned int8 checkpoint — the bf16 file has no int8 weights to keep, so
    there is nothing to choose between.
    """
    if not pruned:
        n, c = plan_vram(free_gb, mp=mp, resident_gb=_RESIDENT_GB, adapter_gb=adapter_gb)
        return "nf4", n, c, "bf16 checkpoint — NF4 is the only option"

    i_swap, i_ckpt = plan_vram(free_gb, mp=mp, resident_gb=_RESIDENT_INT8_GB,
                               transient_gb=_INT8_TRANSIENT_GB, adapter_gb=adapter_gb)
    if i_swap == 0:
        return "int8", i_swap, i_ckpt, "int8 fits with no block swap — the most accurate base"

    n_swap, n_ckpt = plan_vram(free_gb, mp=mp, resident_gb=_RESIDENT_PRUNED_GB,
                               adapter_gb=adapter_gb)
    if n_swap == 0:
        return ("nf4", n_swap, n_ckpt,
                f"int8 would need {i_swap} of 50 blocks on CPU (~4x slower); 4-bit fits entirely "
                f"in VRAM, at ~9% more error in the frozen base")
    return ("nf4", n_swap, n_ckpt,
            f"neither fits outright — 4-bit parks {n_swap} blocks against int8's {i_swap}")


def plan_vram(free_gb: float, mp: float = 0.25, batch: int = 1, resident_gb: float = None,
              transient_gb: float = 0.0, adapter_gb: float = 0.0):
    """Pure planner: (blocks_to_swap, gradient_checkpointing) from free VRAM + token load.

    Token load scales the activation term linearly (tokens ∝ mp x batch). Checkpointing is
    preferred OFF (faster) when everything fits without it; forced ON whenever swap is needed
    (without recompute, autograd would pin every swapped block's weights through backward).
    Swap additionally budgets _SWAP_TRANSIENT_GB: the backward pass transiently holds
    recompute segments beyond the parked residency (measured, see anchors above)."""
    resident = _RESIDENT_GB if resident_gb is None else float(resident_gb)
    # adapter_gb is resident for the whole run (weights + grads + optimizer state), so it belongs
    # in the base, not the activation term — gradient checkpointing does not reduce it.
    base = resident + float(transient_gb) + float(adapter_gb)
    scale = max(0.25, float(mp)) / 0.25 * max(1, int(batch))
    # _NOCKPT_MARGIN_GB, not just _RESERVE_GB: see the note on the constant. Recompute is ~0.1 s
    # a step and worth ~5 GB, so skipping it on a thin margin is a bad trade in both directions.
    need_nockpt = base + _ACT_GB_NOCKPT * scale + _RESERVE_GB + _NOCKPT_MARGIN_GB
    if free_gb >= need_nockpt:
        return 0, False
    need_ckpt = base + _ACT_GB_CKPT * scale + _RESERVE_GB
    if free_gb >= need_ckpt:
        return 0, True
    deficit = need_ckpt + _SWAP_TRANSIENT_GB - free_gb
    blocks = min(40, int(deficit / _PER_BLOCK_GB + 0.999))
    return blocks, True


def is_pruned_checkpoint(path: str) -> bool:
    """Does this file carry the curve-table AdaLN? Reads only the safetensors header.

    Needed before the base loads, because the pruned build's NF4 residency is ~6 GB smaller and
    the swap planner would otherwise park blocks nobody needs parked."""
    import json
    import struct
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return "adaln_t_table" in json.loads(f.read(n))
    except Exception:
        return False


def read_sample_override(output_dir):
    """Live sample override written by the GUI to <output_dir>/.sample_override.json.

    Returns {prompt, seed, width, height} while active, else None. Unlike Krea 2 there is no
    ref_image: H3 is not an edit model, so a reference is meaningless here and a prompt is
    required for the override to count."""
    import json
    path = os.path.join(output_dir, ".sample_override.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        prompt = str(d.get("prompt", "")).strip()
        if not prompt:
            return None
        return {"prompt": prompt,
                "seed": int(d.get("seed", 1234)),
                "width": int(d.get("width", 768)),
                "height": int(d.get("height", 768))}
    except Exception:
        return None


def sample_sigmas(batch: int, device, shift=None, generator=None,
                  image_tokens: int = None) -> torch.Tensor:
    """Noise levels in (0,1) for training.

    shift=None (the default): sigma = 12u/(1+11u), u ~ uniform — H3's OWN training density.
    ai-toolkit's per-model defaults override the global 'sigmoid' with timestep_type='shift'
    through a scheduler configured shift=12 (ui options.tsx + their scheduler_config), so this
    is what MiniMax LoRAs are actually trained with there: median sigma ~0.92, ~57% of steps
    above 0.9, ~3% below 0.3. Training lives at the high-noise end, where each step nudges
    broad structure gently — which is why 1e-4 is a sane LR there and scorching at low shifts.
    (An earlier run here blamed shift-12 for poor likeness; that verdict was confounded —
    bf16 adaln was eating half the LoRA and being dropped at inference, and the pack had no
    audio rows yet. Withdrawn.)

    shift="sigmoid": UNSHIFTED logit-normal, sigma = sigmoid(N(0,1)), median 0.5 — the
    SD3/Flux-style density (ai-toolkit's GLOBAL default, but NOT its MiniMax one). Trains the
    mid/low-noise zone hard: at 1e-4 a 46-image epoch visibly overdrove the adapters
    (real-run finding, twice). A/B use only.

    shift="resolution": logit-normal with a resolution-dependent shift (~1.7 @768^2, median
    0.62 — Krea 2's mapping). Fizgig's original replacement density; same overdrive failure.

    shift=<float>: the uniform-u + shift map at any other value.
    """
    if shift is None:
        shift = VIDEO_SIGMA_SHIFT_TRAIN
    if shift == "sigmoid":
        return torch.sigmoid(torch.randn(batch, device=device, generator=generator))
    if shift == "resolution":
        tokens = float(image_tokens or 225)                       # ~0.25 MP default
        mu = 0.5 + (tokens - 256.0) * (1.15 - 0.5) / (6400.0 - 256.0)
        s = math.exp(mu)
        base = torch.sigmoid(torch.randn(batch, device=device, generator=generator))
    elif isinstance(shift, str) and shift.startswith("lognorm:"):
        # SHAPE, not amount. Same shift map, but a logit-normal base instead of a uniform one:
        # the mass piles up in the middle and thins at BOTH ends, where a uniform base has fat
        # tails. Krea 2 and Klein both draw logit-normal, so this is the one axis the numeric
        # ladder cannot reach — it only ever varies how much low-noise training there is, never
        # where the rest of the mass sits.
        s = float(shift.split(":", 1)[1])
        base = torch.sigmoid(torch.randn(batch, device=device, generator=generator))
    else:
        s = float(shift)
        base = torch.rand(batch, device=device, generator=generator)
    return (s * base) / (1.0 + (s - 1.0) * base)


def compute_loss(model, latent: torch.Tensor, text_embeds: torch.Tensor, *,
                 sigma: torch.Tensor = None, shift: float = None, generator=None,
                 noise: torch.Tensor = None):
    """One image-training step's loss.

    latent      : [1, 24, 1, H, W] clean VAE latent (x0).
    text_embeds : [1, L, text_dim] Qwen3-VL states.
    noise       : optional fixed noise (reproducible steps / tests); else sampled.
    Returns (loss, sigma_used) — MSE of the DiT's video_out against (x0 - noise).
    """
    if latent.shape[0] != 1:
        raise ValueError("MiniMax H3 image training is batch size 1")
    device = latent.device
    x0 = latent.float()
    # The DiT patchifies with patch_size (1, ph, pw), so the latent's H and W must be divisible by
    # the spatial patch. The dataset buckets on a 16-px step and the VAE is 16x, so a latent can be
    # odd (e.g. a 496-px bucket -> 31-px latent, not divisible by 2). Crop to the patch multiple
    # (drops at most one latent row/col = <=16 px of image edge) so patchify is exact and the target
    # (x0 - noise) stays the same shape as the model's prediction.
    _pt, _ph, _pw = getattr(model, "patch_size", (1, 2, 2))
    _H, _W = x0.shape[-2], x0.shape[-1]
    _Hc, _Wc = (_H // _ph) * _ph, (_W // _pw) * _pw
    if (_Hc, _Wc) != (_H, _W):
        x0 = x0[..., :_Hc, :_Wc].contiguous()
    if noise is None:
        noise = torch.randn(x0.shape, device=device, generator=generator, dtype=torch.float32)
    else:
        noise = noise.to(device=device, dtype=torch.float32)[..., :x0.shape[-2], :x0.shape[-1]]
    if sigma is None:
        # Resolution-aware auto schedule: token count from the (cropped) latent's patch grid.
        _tokens = (x0.shape[-2] // _ph) * (x0.shape[-1] // _pw)
        sigma = sample_sigmas(1, device, shift=shift, generator=generator, image_tokens=_tokens)
    s = sigma.reshape(1, 1, 1, 1, 1).to(torch.float32)

    noised = (1.0 - s) * x0 + s * noise
    t = (1.0 - sigma).to(device)
    pred = model(noised.to(latent.dtype), t, text_embeds)
    target = (x0 - noise).to(pred.dtype)
    return F.mse_loss(pred.float(), target.float()), float(sigma.reshape(-1)[0])


@contextlib.contextmanager
def lora_disabled(network):
    """Run the frozen BASE inside this block — every adapter's multiplier is temporarily 0.

    Every module type (LoRA, LoKR, LoHa) reads self.multiplier live in its forward and
    short-circuits on 0.0, so this needs no re-apply and no weight surgery. Restores whatever
    each module had, not a blanket 1.0 — a context LoRA rides at its own strength."""
    mods = list(getattr(network, "unet_loras", []))
    saved = [m.multiplier for m in mods]
    try:
        for m in mods:
            m.multiplier = 0.0
        yield
    finally:
        for m, v in zip(mods, saved):
            m.multiplier = v


def compute_distill_loss(model, network, latent, text_plain, *, text_ref, ref_latents,
                         text_token_tags=None, distill_weight=0.8, shift=None, generator=None,
                         noise=None, seed=0, parts_out=None):
    """Reference distillation: teach the LoRA to behave, from text alone, as if it had been
    shown the reference photo.

    Two predictions of the SAME noised latent at the SAME timestep:
      teacher — frozen base, LoRA off, conditioning WITH the reference (vision blocks + ref rows)
      student — LoRA on, conditioning WITHOUT it
    loss = w * MSE(student, teacher) + (1 - w) * MSE(student, x0 - noise)

    The photo term is what keeps real photographic detail available: pure distillation caps the
    LoRA at exactly the teacher's habits and can never exceed them. The teacher term is what
    stops the run spending capacity on backgrounds and framing, because the target is no longer
    a particular photograph.

    Everything the two passes share is drawn ONCE — noise, timestep, and the audio silence rows.
    The audio rows especially: model.forward redraws them per call when not given, so letting
    each pass draw its own would put a different soundtrack under teacher and student and add
    pure noise to the very signal being distilled.
    """
    if latent.shape[0] != 1:
        raise ValueError("MiniMax H3 image training is batch size 1")
    device = latent.device
    x0 = latent.float()
    _pt, _ph, _pw = getattr(model, "patch_size", (1, 2, 2))
    _H, _W = x0.shape[-2], x0.shape[-1]
    _Hc, _Wc = (_H // _ph) * _ph, (_W // _pw) * _pw
    if (_Hc, _Wc) != (_H, _W):
        x0 = x0[..., :_Hc, :_Wc].contiguous()
    if noise is None:
        noise = torch.randn(x0.shape, device=device, generator=generator, dtype=torch.float32)
    else:
        noise = noise.to(device=device, dtype=torch.float32)[..., :x0.shape[-2], :x0.shape[-1]]

    _tokens = (x0.shape[-2] // _ph) * (x0.shape[-1] // _pw)
    sigma = sample_sigmas(1, device, shift=shift, generator=generator, image_tokens=_tokens)
    s = sigma.reshape(1, 1, 1, 1, 1).to(torch.float32)
    noised = ((1.0 - s) * x0 + s * noise).to(latent.dtype)
    t = (1.0 - sigma).to(device)

    # one soundtrack for both passes (see the docstring)
    audio_noise = None
    if getattr(model, "pack_audio_rows", False):
        from fizgig.minimax.model import AUDIO_CHANNELS, audio_latents_for_frames
        n_a = audio_latents_for_frames(1) * AUDIO_CHANNELS
        audio_noise = torch.randn(n_a, model.config.audio_latents_dim, device=device,
                                  generator=generator, dtype=torch.float32)

    with torch.no_grad(), lora_disabled(network):
        teacher = model(noised, t, text_ref, audio_noise, ref_latents=ref_latents,
                        text_token_tags=text_token_tags, seed=seed).float()
    student = model(noised, t, text_plain, audio_noise).float()

    w = float(distill_weight)
    teacher_mse = F.mse_loss(student, teacher.detach())
    loss = w * teacher_mse
    photo_mse = None
    if w < 1.0:
        photo_mse = F.mse_loss(student, (x0 - noise).float())
        loss = loss + (1.0 - w) * photo_mse
    if parts_out is not None:
        # The RAW errors, before the 0.8/0.2 weights. The weights are already known; what is not
        # is how BIG each error is — and "how much of the learning comes from real pixels" is a
        # question about the errors, not the weights. Matching a real photograph is harder than
        # matching the model's own output, so the photo term can punch well above its weight.
        parts_out["teacher"] = float(teacher_mse.detach())
        parts_out["photo"] = float(photo_mse.detach()) if photo_mse is not None else 0.0
    return loss, float(sigma.reshape(-1)[0])


# ---------------------------------------------------------------------------
# Adaptive LR — bi-directional plateau tracker (architecture-agnostic; a faithful port of the
# Klein/Krea 2 watcher). Stability signal is weight-norm growth (>30%), same as Krea 2 (the H3
# loop clips gradients but the watcher reads weight-norm growth, not the clip ratio).
# ---------------------------------------------------------------------------
class AdaptiveLR:
    """Each epoch boundary: probe UP x1.25 on steady loss descent (patience 2); reduce DOWN x0.5
    on loss plateau (patience ramp) or a stability signal. On a stability event it blends the LoRA
    weights 70/30 toward the previous epoch's snapshot and restores the optimizer state (kills bad
    Adam momentum). The CPU rollback snapshot is in-memory only; the streak/best_loss scalars are
    JSON round-trippable (kept for parity — this barebones trainer has no resume yet)."""

    BLEND = 0.7
    WEIGHT_GROWTH_THRESHOLD = 0.30

    def __init__(self, min_lr, max_lr):
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.best_loss = None
        self.good_streak = 0
        self.bad_streak = 0
        self.stability_streak = 0
        self.stability_triggered = False
        self.prev_weight_norm = None
        self.snapshot = None  # {"weights": {...cpu...}, "optim": cpu state} — not persisted

    def state_dict(self):
        return {"best_loss": self.best_loss, "good_streak": self.good_streak,
                "bad_streak": self.bad_streak, "stability_streak": self.stability_streak,
                "stability_triggered": self.stability_triggered,
                "prev_weight_norm": self.prev_weight_norm}

    def load_state_dict(self, d):
        if not d:
            return
        self.best_loss = d.get("best_loss")
        self.good_streak = int(d.get("good_streak", 0))
        self.bad_streak = int(d.get("bad_streak", 0))
        self.stability_streak = int(d.get("stability_streak", 0))
        self.stability_triggered = bool(d.get("stability_triggered", False))
        self.prev_weight_norm = d.get("prev_weight_norm")

    @staticmethod
    def _weight_norm(network):
        wn = 0.0
        with torch.no_grad():
            for p in network.parameters():
                if p.requires_grad:
                    wn += float(p.detach().float().norm().item()) ** 2
        return wn ** 0.5

    def _snapshot(self, network, optimizer):
        with torch.no_grad():
            weights = {n: p.detach().clone().to("cpu")
                       for n, p in network.named_parameters() if p.requires_grad}

        def _cpu(o):
            if isinstance(o, torch.Tensor):
                return o.detach().clone().to("cpu")
            if isinstance(o, dict):
                return {k: _cpu(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_cpu(v) for v in o]
            return o
        try:
            self.snapshot = {"weights": weights, "optim": _cpu(optimizer.state_dict())}
        except Exception:
            self.snapshot = {"weights": weights, "optim": None}

    def _rollback(self, network, optimizer):
        cur = dict(network.named_parameters())
        with torch.no_grad():
            for name, prev in self.snapshot["weights"].items():
                if name in cur and cur[name].requires_grad:
                    p = cur[name]
                    prev_d = prev.to(device=p.device, dtype=p.dtype)
                    p.copy_(self.BLEND * prev_d + (1.0 - self.BLEND) * p)
        if self.snapshot.get("optim") is not None:
            try:
                optimizer.load_state_dict(self.snapshot["optim"])
            except Exception:
                pass

    def epoch_boundary(self, epoch, current_loss, network, optimizer):
        """epoch is 0-indexed (global). epoch 0 arms the baseline; epoch >= 1 adjusts the LR."""
        if epoch == 0:
            self.best_loss = current_loss
            self.prev_weight_norm = self._weight_norm(network)
            logger.info(f"[adaptive_lr] epoch 1: loss={current_loss:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} | ARMED")
            self._snapshot(network, optimizer)
            return

        patience_up = 2
        patience_down = 2 if (self.stability_triggered or epoch == 1 or epoch >= 4) else 1
        cur_lr = optimizer.param_groups[0]["lr"]
        new_lr = cur_lr
        cur_wn = self._weight_norm(network)
        weight_growth = None
        if self.prev_weight_norm and self.prev_weight_norm > 0:
            weight_growth = (cur_wn - self.prev_weight_norm) / self.prev_weight_norm
        stability_reason = None
        if weight_growth is not None and weight_growth > self.WEIGHT_GROWTH_THRESHOLD:
            stability_reason = f"wnorm_Δ {weight_growth*100:+.0f}% > {self.WEIGHT_GROWTH_THRESHOLD*100:.0f}%"

        action, reason = "HOLD", ""
        if stability_reason is not None:
            self.stability_streak += 1
            stability_patience = 1 if not self.stability_triggered else 2
            if self.stability_streak >= stability_patience:
                candidate = max(cur_lr * 0.5, self.min_lr)
                note = ""
                if self.snapshot is not None:
                    self._rollback(network, optimizer)
                    note = f"; blended {int(self.BLEND*100)}/{int((1-self.BLEND)*100)} + optim restored"
                if candidate < cur_lr:
                    new_lr = candidate
                    action = "REDUCE+ROLLBACK" if self.snapshot is not None else "REDUCE"
                else:
                    action = "HOLD (floored)"
                reason = f"stability: {stability_reason}{note}"
                self.good_streak = self.bad_streak = self.stability_streak = 0
                self.stability_triggered = True
            else:
                action = "WAIT"
                reason = f"stability: {stability_reason}, streak {self.stability_streak}/{stability_patience}"
        elif self.best_loss is None or current_loss < self.best_loss:
            self.stability_streak = 0
            self.best_loss = current_loss
            self.good_streak += 1
            self.bad_streak = 0
            if self.good_streak >= patience_up:
                candidate = min(cur_lr * 1.25, self.max_lr)
                if candidate > cur_lr:
                    new_lr = candidate
                    action = "PROBE UP"
                    reason = f"loss improving, streak {self.good_streak}"
                else:
                    action = "HOLD (capped)"
                    reason = "loss improving, at max_lr"
                self.good_streak = 0
            else:
                reason = f"loss improving, streak {self.good_streak}/{patience_up}"
        else:
            self.stability_streak = 0
            self.bad_streak += 1
            self.good_streak = 0
            if self.bad_streak >= patience_down:
                candidate = max(cur_lr * 0.5, self.min_lr)
                if candidate < cur_lr:
                    new_lr = candidate
                    action = "REDUCE"
                    reason = f"loss plateau, streak {self.bad_streak}"
                else:
                    action = "HOLD (floored)"
                    reason = "loss plateau, at min_lr"
                self.bad_streak = 0
            else:
                reason = f"loss plateau, streak {self.bad_streak}/{patience_down}"

        if new_lr != cur_lr:
            # Respect a depth-split LR: each group carries its own lr_scale, so the watcher moves
            # the whole schedule up or down while KEEPING the ratio between groups. Writing new_lr
            # flat would silently undo the split on the first adaptive move.
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr * pg.get("lr_scale", 1.0)
        lr_str = f"{cur_lr:.2e}" if new_lr == cur_lr else f"{cur_lr:.2e}->{new_lr:.2e}"
        wn_str = f"{weight_growth*100:+.0f}%" if weight_growth is not None else "—"
        logger.info(f"[adaptive_lr] epoch {epoch + 1}: loss={current_loss:.4f} lr={lr_str} "
                    f"wnorm_Δ={wn_str} | {action} ({reason})")
        self.prev_weight_norm = cur_wn
        self._snapshot(network, optimizer)


# ---------------------------------------------------------------------------
# Per-block movement limiter — a compressor on the block bus.
#
# Empirical finding (8 Aug, three runs + a block-range A/B): whichever block sits LAST in the
# trained range absorbs wildly disproportionate movement — 2-4x the median block from epoch 1,
# and still diverging 40 epochs later. Cut blocks 46-49 and blocks 43-45 inherit the exact
# same signature: the pathology is POSITIONAL, not a property of particular layers. The
# deepest trained block gets the most coherent, least-attenuated gradient (everything after
# it is frozen and decorrelates nothing), and Adam turns coherence into relentless movement.
# The visible symptom is output-adjacent over-editing: distorted eyes and other
# high-frequency damage.
#
# LR penalties and block cuts are positional patches for a positional problem — they just
# relocate the hot spot. This limiter is self-targeting: after each optimizer step, any
# block whose TOTAL RELATIVE movement (sum over its adapters of ||dW||/||W_base||, the same
# metric the offline analysis used) exceeds `cap_factor x median block` is projected back to
# the cap by scaling its up-factors. Blocks move freely until one hogs; then only that one
# is pulled back, wherever the trained range ends.
# ---------------------------------------------------------------------------
class StepClipper:
    """Cap how far any block may move in a SINGLE optimizer step.

    Replaces the cumulative BlockLimiter that shipped in 3.5.0. That one clamped a block's
    TOTAL accumulated movement back to cap x median, which necessarily scaled down everything
    the block had legitimately learned in earlier epochs along with the overshoot — measured on
    real runs as a genuine likeness ceiling: limiter ON was visibly worse than OFF, while OFF
    corrupted. Clipping the STEP prevents the overshoot instead of undoing the history, so
    there is no quality to trade for the safety.

    Being per-step also removes the calibration problem that sank the movement governor: a
    per-epoch budget has to be scaled by dataset size, and got it wrong by 7x on a 272-step
    epoch, starving a run for 84 epochs. A step is a step on any dataset.

    Measured in MODEL space — the change this step in each block's effective delta, summed in
    quadrature across the block's modules — and cheap, because every term is a rank-sized
    product via <kron(a,b), kron(c,d)> = <a,c><b,d> and <UV, XY> = tr(U^T X Y V^T). A full
    weight matrix is never materialised. Over-cap blocks are lerped back toward their pre-step
    weights, which is exact to first order in the step size (the delta is bilinear in the
    factors, so the second-order term is negligible at real step sizes).

    Self-calibrating: the cap is a multiple of the MEDIAN block's step, so it needs no absolute
    threshold and targets whichever block is actually running hot — the caboose, wherever the
    trained range happens to end.
    """

    def __init__(self, network, cap_factor: float = 1.25):
        import re as _re
        self.cap = float(cap_factor)
        self.clamped_total = 0
        self.clamp_counts = {}
        self.groups = {}              # block id -> [module]
        for m in getattr(network, "unet_loras", []):
            blk = _re.search(r"blocks_(\d+)_", m.lora_name)
            if blk is None or "token_refiner" in m.lora_name:
                continue              # text-side refiner is not part of the depth argument
            self.groups.setdefault(int(blk.group(1)), []).append(m)
        # Pre-step parameter snapshot, allocated ONCE and copied into each step.
        self._params = {blk: [p for m in mods for p in m.parameters() if p.requires_grad]
                        for blk, mods in self.groups.items()}
        self._prev = {blk: [p.detach().clone() for p in ps] for blk, ps in self._params.items()}
        self._prev_f = {}             # per-module factor snapshot for the delta measurement
        # Clip on each block's SMOOTHED step rate, not its instantaneous step. The caboose is a
        # PERSISTENTLY hot block; a single step landing above the median is just noise, and
        # per-step movement is far noisier than the cumulative quantity the retired limiter
        # measured. Reusing 1.25x on the raw per-step value therefore braked whichever blocks
        # were learning fastest on any given step — measured as a real quality loss that no LR
        # change touched (halving the LR moved the dose by 8%).
        self._rate = {}               # block id -> EMA of its per-step movement
        self._clipped_steps = 0
        self._total_steps = 0
        self._tail = 0.0              # last measured peak/median ACCUMULATED movement

    @staticmethod
    def _factors(m):
        """(a, b, scale) such that the module's delta is scale * a (x) b, for whichever form."""
        if hasattr(m, "lokr_w1"):
            return m.lokr_w1, m.lokr_w2, 1.0        # Fizgig LoKR: alpha 1.0, scale 1.0
        return m.lora_up.weight, m.lora_down.weight, float(m.scale)

    @classmethod
    def _cum_sq(cls, m) -> float:
        """||D||_F^2 for one module — its ACCUMULATED delta, not this step's."""
        a, b, sc = cls._factors(m)
        a, b = a.float(), b.float()
        if hasattr(m, "lokr_w1"):
            n = (a.norm() * b.norm()) ** 2
        else:
            n = torch.trace((a.T @ a) @ (b @ b.T)).clamp(min=0)
        return float(n) * sc * sc

    @classmethod
    def _step_delta_sq(cls, m, prev) -> float:
        """||D_post - D_pre||_F^2 for one module, without materialising D."""
        a1, b1, sc = cls._factors(m)
        a0, b0 = prev
        a1, b1, a0, b0 = a1.float(), b1.float(), a0.float(), b0.float()
        if hasattr(m, "lokr_w1"):
            # <kron(a,b), kron(c,d)> = <a,c><b,d>
            n1 = (a1.norm() * b1.norm()) ** 2
            n0 = (a0.norm() * b0.norm()) ** 2
            cross = (a1 * a0).sum() * (b1 * b0).sum()
        else:
            # <U1 V1, U0 V0> = tr(U1^T U0 V0 V1^T); ||UV||^2 = tr((U^T U)(V V^T))
            n1 = torch.trace((a1.T @ a1) @ (b1 @ b1.T))
            n0 = torch.trace((a0.T @ a0) @ (b0 @ b0.T))
            cross = torch.trace((a1.T @ a0) @ (b0 @ b1.T))
        return float((n1 + n0 - 2 * cross).clamp(min=0)) * sc * sc

    @torch.no_grad()
    def pre_step(self):
        """Snapshot the weights the optimizer is about to move. Call BEFORE optimizer.step()."""
        for blk, ps in self._params.items():
            for dst, p in zip(self._prev[blk], ps):
                dst.copy_(p.detach())
        self._prev_f = {id(m): tuple(t.detach().clone() for t in self._factors(m)[:2])
                        for mods in self.groups.values() for m in mods}

    @torch.no_grad()
    def step(self):
        """Clip blocks whose SMOOTHED movement rate is running above cap x the pack's."""
        import statistics as _st
        if len(self.groups) < 3 or not self._prev_f:
            return
        moved = {blk: sum(self._step_delta_sq(m, self._prev_f[id(m)]) for m in mods) ** 0.5
                 for blk, mods in self.groups.items()}
        for blk, d in moved.items():
            r = self._rate.get(blk)
            self._rate[blk] = d if r is None else 0.9 * r + 0.1 * d
        med = _st.median(self._rate.values())
        self._total_steps += 1
        if med <= 0:
            return                                  # nothing has moved yet
        cap = self.cap * med
        # ACCUMULATION AWARENESS. Capping strides bounds how fast a block moves but not how far
        # it has GOT — and a coherent run (which is what gradient accumulation produces) lets
        # the caboose accumulate imbalance even while every stride is legal: measured at 2.02x
        # the median block by epoch 2 with strides capped at 1.25x. The old limiter fixed that
        # by scaling the block's accumulated delta down, which also destroyed what it had
        # legitimately learned. Instead, a block that is ALREADY ahead simply gets a tighter
        # step allowance until the pack catches up: no history is ever touched, the block just
        # stops pulling further away. Squeeze is proportional and floored so it never freezes.
        cums = {blk: sum(self._cum_sq(m) for m in mods) ** 0.5
                for blk, mods in self.groups.items()}
        med_cum = _st.median(cums.values())
        self._tail = (max(cums.values()) / med_cum) if med_cum > 0 else 0.0
        _fired = False
        for blk, d in moved.items():
            blk_cap = cap
            if med_cum > 0 and cums[blk] > self.cap * med_cum:
                # SQRT, not the raw ratio, and floored at 0.5. An already-ahead block is
                # otherwise penalised twice over — once by the per-step cap for being above the
                # median, again by this squeeze for being ahead — and those are the same late
                # blocks every time, so the stacked penalty reads as a treble cut. At a 2.41x
                # tail under a 2.0 cap the raw ratio pulled the effective cap down to 1.66;
                # softened it is 1.82, so raising the cap actually raises it. Genuine runaways
                # still get squeezed, just proportionally less hard.
                blk_cap = cap * max(0.5, ((self.cap * med_cum) / cums[blk]) ** 0.5)
            # TREND decides whether to act — that is what makes a persistently hot block (the
            # caboose) the target and lets a one-off noisy step from a healthy block through.
            # The TRIM is then applied to this actual step, not to the lagging average: scaling
            # by cap/rate under-corrects badly (a 10x hog only came back to ~5.9x).
            if self._rate[blk] <= blk_cap or d <= blk_cap:
                continue
            cap_ = blk_cap
            s = cap_ / d
            for p, prev in zip(self._params[blk], self._prev[blk]):
                p.data.lerp_(prev, 1.0 - s)
            # The trend must reflect what actually happened, not the pre-trim step, or it stays
            # inflated and keeps re-triggering on a block that is now behaving.
            self._rate[blk] -= 0.1 * (d - cap_)
            self.clamped_total += 1
            self.clamp_counts[blk] = self.clamp_counts.get(blk, 0) + 1
            _fired = True
        if _fired:
            self._clipped_steps += 1

    def epoch_report(self):
        # The clip-rate is the number that matters as much as WHICH blocks: a cap that fires on
        # most steps is braking the whole pack, not trimming a caboose, and that reads from the
        # outside as "quality is worse" with no distortion to point at. A healthy run trims a
        # few persistent blocks; if this says most steps, the cap is too tight for the dataset.
        pct = (100.0 * self._clipped_steps / self._total_steps) if self._total_steps else 0.0
        self._clipped_steps = self._total_steps = 0
        tail = f" · tail {self._tail:.2f}x median" if self._tail else ""
        if not self.clamp_counts:
            return f"[clip] no block ran above the cap this epoch{tail}"
        top = sorted(self.clamp_counts.items(), key=lambda kv: -kv[1])[:6]
        n_blocks = len(self.clamp_counts)
        self.clamp_counts = {}
        return (f"[clip] fired on {pct:.0f}% of steps across {n_blocks} block(s){tail} — "
                + ", ".join(f"block {b} x{n}" for b, n in top))


class BlockLimiter:
    """RETIRED (10 Aug) — kept only because the offline analysis scripts import _movement.

    Superseded by StepClipper: clamping CUMULATIVE movement also scaled down legitimately
    learned history, which measurably capped likeness. Do not wire this into the loop."""

    def __init__(self, network, dit, cap_factor: float = 1.5):
        import re as _re
        self.cap = float(cap_factor)
        self.clamped_total = 0
        self.clamp_counts = {}
        # Own the module map, by ISINSTANCE — the shared _build_dit_linear_map filters on the
        # exact class NAME "Linear", which made the int8 base's ConvRotInt8Linear (and bnb's
        # Linear4bit) invisible: on a real base the limiter watched ZERO blocks and reported
        # "no block exceeded the cap" while the caboose ran hot. Both are nn.Linear subclasses.
        linear_map = {"lora_unet_" + n.replace(".", "_"): m
                      for n, m in dit.named_modules() if isinstance(m, torch.nn.Linear)}
        self.groups = {}          # block id -> [(module, base_norm)]
        for m in getattr(network, "unet_loras", []):
            blk = _re.search(r"blocks_(\d+)_", m.lora_name)
            if blk is None or "token_refiner" in m.lora_name:
                continue          # text-side refiner is not part of the depth argument
            target = linear_map.get(m.lora_name)
            bn = self._base_norm(target) if target is not None else None
            if bn:
                self.groups.setdefault(int(blk.group(1)), []).append((m, bn))

    @staticmethod
    def _base_norm(mod) -> float:
        """||W_base||_F for whichever storage the base uses. The ConvRot rotation is
        orthogonal, so the norm of the int8 codes x scales IS the true weight norm."""
        import torch as _t
        with _t.no_grad():
            if hasattr(mod, "qdata"):                        # ConvRotInt8Linear
                return float((mod.qdata.float() * mod.wscale.float()).norm())
            w = getattr(mod, "weight", None)
            if w is None:
                return 0.0
            if w.__class__.__name__ == "Params4bit":         # bnb NF4 shell
                try:
                    import bitsandbytes.functional as _bf
                    return float(_bf.dequantize_4bit(w.data, w.quant_state).float().norm())
                except Exception:
                    return 0.0
            return float(w.float().norm())

    @staticmethod
    def _movement(m) -> float:
        """||dW||_F for one adapter, exactly as the offline analysis computes it."""
        import torch as _t
        with _t.no_grad():
            if hasattr(m, "lokr_w1"):
                return float(m.lokr_w1.float().norm() * m.lokr_w2.float().norm()) * float(m.scale)
            up, dn = m.lora_up.weight.float(), m.lora_down.weight.float()
            g = _t.trace((up.T @ up) @ (dn @ dn.T)).clamp(min=0)
            return float(g.sqrt()) * float(m.scale)

    @torch.no_grad()
    def step(self):
        """Project any over-cap block back to cap_factor x median. Cheap: rank-sized matmuls.

        The metric is RAW ||dW|| per block — NOT movement relative to the block's base norm.
        It used to be relative, and a real run showed why that leaks: damage correlates with
        raw delta (the whole dose-response table is in raw units), late blocks have LARGER
        base weights, so the relative metric granted the tail extra raw allowance — while
        the limiter held every block to 1.25x in ITS units, the tail crept to 2.34x median
        in raw units, over the damage threshold the governor was holding the pack under.
        All H3 blocks are identical shapes, so raw norms are directly comparable."""
        import statistics as _st
        rel = {}
        for blk, mods in self.groups.items():
            rel[blk] = sum(self._movement(m) for m, _bn in mods)
        if len(rel) < 3:
            return
        med = _st.median(rel.values())
        if med <= 0:
            return                                            # nothing has moved yet
        cap = self.cap * med
        for blk, r in rel.items():
            if r <= cap:
                continue
            s = cap / r
            for m, _bn in self.groups[blk]:
                if hasattr(m, "lokr_w2"):
                    m.lokr_w2.mul_(s)                         # delta scales linearly in w2
                else:
                    m.lora_up.weight.mul_(s)
            self.clamped_total += 1
            self.clamp_counts[blk] = self.clamp_counts.get(blk, 0) + 1

    def epoch_report(self):
        if not self.clamp_counts:
            return "[limiter] no block exceeded the cap this epoch"
        top = sorted(self.clamp_counts.items(), key=lambda kv: -kv[1])[:6]
        msg = "[limiter] clamped " + ", ".join(f"block {b} x{n}" for b, n in top)
        self.clamp_counts = {}
        return msg


class AdapterRamp:
    """Hold each step at a constant FRACTION of the adapter's current size, ramping the LR up
    toward the configured ceiling as the adapter grows.

    The observation this comes from: an adapter at ||dW|| ~53, trained slowly for 92 epochs,
    took a full 2e-4 for ten epochs with no distortion at all and produced the best likeness of
    the project. A fresh adapter at ||dW|| ~3 is visibly damaged by half that. The rate was
    never the problem — the SAME step is a 9% perturbation of a mature adapter and a 150%
    perturbation of a new one. A LoRA starts at exactly zero, so the ratio of step size to
    adapter size is at its worst on step one and improves monotonically from there.

    Which means the conventional schedule is backwards for adapters. Warmup-then-decay is built
    for models that start from a sensible initialisation; here it is too hot when the adapter is
    tiny and too cold once the adapter could take it. This ramps the other way.

    Why it needs no calibration, unlike the retired movement governor: the governor servoed on
    an ABSOLUTE movement rate, which depends on dataset size, network type and model width — it
    was wrong by 7x on a 272-step epoch. `step / ||dW||` is dimensionless, so one target
    transfers across datasets, LoRA vs LoKR, and any model size.

    At equilibrium the adapter grows exponentially (d||dW||/dt = rho*||dW||) until the LR hits
    the ceiling, after which growth returns to linear. rho is therefore best read as a growth
    rate: 0.005/step doubles the adapter roughly every 140 steps."""

    def __init__(self, network, target_rel: float = 0.005, start_mult: float = 0.1):
        self.target = float(target_rel)
        self.mult = float(start_mult)
        self._smooth = None
        self._prev = None
        self.params = [p for p in network.parameters() if p.requires_grad]
        self._mods = [m for m in getattr(network, "unet_loras", [])]

    @torch.no_grad()
    def _size(self) -> float:
        """||dW|| across the whole adapter — model-space, not parameter-space."""
        return sum(StepClipper._cum_sq(m) for m in self._mods) ** 0.5

    @torch.no_grad()
    def step(self) -> float:
        cur = self._size()
        if self._prev is None or cur <= 1e-9:
            self._prev = cur
            return self.mult
        rel = max(0.0, cur - self._prev) / cur      # this step as a fraction of what exists
        self._prev = cur
        self._smooth = rel if self._smooth is None else 0.9 * self._smooth + 0.1 * rel
        if self._smooth > 1e-12:
            err = self._smooth / self.target
            # Per-step gain caps, both damped after a real run hunted and then DAMAGED the
            # model on the way back up: 22 -> 77 -> 73 -> 68 -> 63 -> 29 -> 100 across
            # consecutive epochs, and the jump to 100% hit an adapter that was not ready for
            # it. The RELEASE rate is therefore a safety parameter in its own right, not a
            # tuning nicety — the old 1.03 compounds to 3.9x over a 46-step epoch, enough to
            # go from a third of the ceiling to all of it in one epoch. 1.01 caps that at
            # ~1.6x per epoch, so the ceiling is approached over several epochs and the
            # adapter has time to grow into it.
            #
            # The old 0.70 down cap compounds to 4e-8 over the same epoch — a 12:1 asymmetry
            # against the up-gain that caused the slam-to-floor half of the oscillation, whose
            # rebound was what overshot. 0.95 keeps a safety bias (still ~5x faster down than
            # up) without flooring the LR from a single noisy reading.
            self.mult = min(1.0, max(0.02, self.mult * min(1.01, max(0.95, err ** -0.3))))
        return self.mult

    def epoch_report(self) -> str:
        rel = (self._smooth or 0.0)
        return (f"[ramp] adapter ||dW||={self._prev or 0:.2f}, growing {100 * rel:.3f}%/step "
                f"(target {100 * self.target:.3f}%) — LR at {100 * self.mult:.0f}% of the "
                f"configured ceiling")


def should_reassert_lr(*, resuming, adaptive, ramp, warmup_steps, global_step) -> bool:
    """Does anything write param_group['lr'] from here on? If not, a resume must reassert the
    CONFIGURED rate.

    torch's optimizer.load_state_dict restores the saved param_groups INCLUDING lr, and the
    step loop only writes lr while warmup is still ramping. A state written while something
    WAS modulating the LR (the retired movement governor throttled it) therefore handed its
    last throttled rate to a run that no longer modulates anything, and kept it for the whole
    run — measured on a real run as 3.28e-5 against a configured 2e-4.

    The subtle case, and the one the first version of this fix got wrong: warmup CONFIGURED but
    already FINISHED. warmup_steps > 0 is not the question — `global_step < warmup_steps` is."""
    if not resuming:
        return False
    if adaptive is not None:
        return False        # adaptive owns the LR; its restored mid-flight value is correct
    if ramp is not None:
        return False        # the adapter ramp rewrites lr every step
    if warmup_steps and global_step < warmup_steps:
        return False        # the warmup ramp rewrites lr every step until it ends
    return True


class EMAWeights:
    """Exponential moving average of the trainable adapter — the smooth center of a rough
    trajectory.

    High static LRs take big Adam strides that zigzag around the good solution; the raw
    weights at any single step are one corner of the zigzag, and that roughness reads as
    distortion in samples. The EMA is the running average of the path, so what gets SAVED
    (and previewed) is the center the strides orbit — the standard diffusion-training cure
    for exactly this. Training itself always runs on the raw weights: swap_in/swap_out
    bracket saves and previews only.

    Decay ramps in as min(decay, (1+n)/(10+n)) so the first steps track the weights closely
    instead of anchoring to the zero init. Shadow is fp32 (the adapter is small)."""

    def __init__(self, network, decay: float):
        self.decay = float(decay)
        self.n = 0
        self.params = [p for p in network.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone().float() for p in self.params]
        self._backup = None

    @torch.no_grad()
    def update(self):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self):
        """Put the averaged weights into the live network (for a save or a preview).

        The raw-weight backup lives on CPU: swap_in brackets previews, and a clip preview
        is exactly when GPU headroom is scarcest — a GPU-resident backup (~0.6 GB at LoKR
        factor 8) was part of what tipped 32 GB cards back into the Windows VRAM spill."""
        self._backup = [p.detach().to("cpu", copy=True) for p in self.params]
        for s, p in zip(self.shadow, self.params):
            p.data.copy_(s.to(p.device, p.dtype))

    @torch.no_grad()
    def swap_out(self):
        """Restore the raw training weights. Must always pair with swap_in."""
        for b, p in zip(self._backup, self.params):
            p.data.copy_(b.to(p.device, p.dtype))
        self._backup = None

    def state_dict(self):
        return {"n": self.n, "decay": self.decay,
                "shadow": [s.detach().cpu() for s in self.shadow]}

    def load_state_dict(self, sd):
        self.n = int(sd["n"])
        if len(sd["shadow"]) != len(self.shadow):
            raise ValueError(f"EMA state has {len(sd['shadow'])} tensors, network has "
                             f"{len(self.shadow)} — different run configuration?")
        self.shadow = [t.to(s.device, torch.float32) for t, s in zip(sd["shadow"], self.shadow)]


# ---------------------------------------------------------------------------
# Full image-only training loop (NF4 base + LoRA) over the H3 caches.
# ---------------------------------------------------------------------------
class _Collator:
    """DataLoader batch_size is always 1 (the dataset batches internally by bucket)."""

    def __init__(self, shared_epoch, dataset):
        self.shared_epoch = shared_epoch
        self.dataset = dataset

    def __call__(self, examples):
        wi = torch.utils.data.get_worker_info()
        ds = wi.dataset if wi is not None else self.dataset
        ds.set_current_epoch(self.shared_epoch.value)
        return examples[0]


def _save_training_state(output_dir, output_name, network, optimizer, *, epoch, global_step,
                         dtype, extra=None, ema=None):
    """Save a resumable training-state dir matching Klein/Krea 2 naming: <name>-<NNNNNN>-state/.

    NNNNNN is the number of COMPLETED epochs (= the next 0-indexed epoch to run). Holds the
    network weights in NATIVE state_dict naming (never the LyCORIS comfy-format rewrite — resume
    load_state_dict needs the module keys), the optimizer state, RNG, and a small JSON. The
    GUI's _detect_latest_state_dir finds the highest-numbered dir and passes it to --resume."""
    import json
    state_dir = os.path.join(output_dir, f"{output_name}-{epoch:06d}-state")
    os.makedirs(state_dir, exist_ok=True)
    try:
        return _write_state_files(state_dir, network, optimizer, epoch=epoch,
                                  global_step=global_step, dtype=dtype, extra=extra, ema=ema)
    except Exception as _first:
        # Clean the partial dir (no training_state.json = no commit marker, but it would shadow
        # the previous good state in the GUI's latest-state scan), then retry ONCE after a short
        # pause. Network filesystems (RunPod volumes) throw transient stream errors that clear
        # in seconds — a real run lost its epoch-8 state to exactly one of those. If the retry
        # also fails it is not transient; re-raise and let the caller decide fatality.
        import shutil
        import time
        shutil.rmtree(state_dir, ignore_errors=True)
        logger.warning("[state] save failed (%s: %s) — retrying once in 5s",
                       type(_first).__name__, _first)
        time.sleep(5)
        try:
            os.makedirs(state_dir, exist_ok=True)
            return _write_state_files(state_dir, network, optimizer, epoch=epoch,
                                      global_step=global_step, dtype=dtype, extra=extra, ema=ema)
        except Exception:
            shutil.rmtree(state_dir, ignore_errors=True)
            raise


def _write_state_files(state_dir, network, optimizer, *, epoch, global_step,
                       dtype, extra=None, ema=None):
    import json
    network.save_weights(os.path.join(state_dir, "lora.safetensors"), dtype,
                         {"ss_architecture": ARCHITECTURE_MINIMAX,
                          "ss_network_module": "fizgig.minimax (state dir, native keys)"})
    torch.save(optimizer.state_dict(), os.path.join(state_dir, "optimizer.pt"))
    if ema is not None:
        # The RAW weights are what lora.safetensors holds (training resumes from them); the
        # EMA shadow rides alongside so the average survives pause/resume too.
        torch.save(ema.state_dict(), os.path.join(state_dir, "ema.pt"))
    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, os.path.join(state_dir, "rng.pt"))
    meta = {"epoch": epoch, "global_step": global_step}
    if extra:
        meta.update(extra)
    with open(os.path.join(state_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    # training_state.json is written LAST on purpose: it is the commit marker. A save that
    # dies partway leaves no json, and both the resume validator and the GUI's latest-state
    # detection treat a json-less dir as not-a-state rather than resuming garbage.
    logger.info(f"[state] saved -> {state_dir}")
    return state_dir


def _validate_state_dir(state_dir):
    """Refuse anything that is not a saved training state, and say what to pick instead.

    Issue #48: choosing the OUTPUT directory rather than a state folder failed with a bare
    "lora.safetensors not found", and the obvious workaround — putting a LoRA there under that
    name — then appeared to work. It cannot: without training_state.json there is no epoch or
    step, and without optimizer.pt there is no Adam state, so the run silently starts over from
    epoch 0 while looking like a resume, and overwrites the finished LoRA on the way. Refusing
    is the only safe answer, and the message has to name the folder they actually wanted.
    """
    if os.path.isfile(state_dir):
        sib = ""
        base = os.path.dirname(state_dir)
        try:
            states = sorted(d for d in os.listdir(base) if d.endswith("-state")
                            and os.path.isfile(os.path.join(base, d, "training_state.json")))
            if states:
                sib = " Next to it: " + ", ".join(states[-3:])
        except OSError:
            pass
        raise RuntimeError(
            f"[resume] {os.path.basename(state_dir)} is a file — resume takes the saved-state "
            f"FOLDER (named like '<lora name>-000012-state'), not a .safetensors.{sib}")
    if not os.path.isdir(state_dir):
        raise RuntimeError(f"[resume] {state_dir} does not exist — was the state folder moved "
                           f"or renamed?")
    missing = [f for f in ("lora.safetensors", "training_state.json")
               if not os.path.isfile(os.path.join(state_dir, f))]
    if not missing:
        return
    lines = [
        f"[resume] {state_dir} is not a saved training state — missing {', '.join(missing)}.",
        "[resume] Pick the folder named like '<lora name>-000012-state'. Renaming a LoRA to "
        "lora.safetensors does not make one: there would be no optimizer state and no epoch "
        "to resume from, so the run would quietly start again from scratch.",
    ]
    try:
        # The usual mistake is picking the parent output directory, one level above the state
        # folders — so if they are sitting right there, name them.
        here = sorted(d for d in os.listdir(state_dir)
                      if d.endswith("-state")
                      and os.path.isfile(os.path.join(state_dir, d, "training_state.json")))
        if here:
            lines.append("[resume] That looks like your output directory. The saved states in "
                         "it are: " + ", ".join(here[-5:]))
    except OSError:
        pass
    raise RuntimeError(os.linesep.join(lines))


def _load_training_state(state_dir, network, optimizer, *, device):
    """Restore network + optimizer + RNG from a state dir. Returns (start_epoch, global_step, meta)."""
    _validate_state_dir(state_dir)
    import json
    from safetensors.torch import load_file
    # strict=False tolerates benign key drift, but if NOTHING matched the network silently stays
    # at its zero init and the run "succeeds" while training from scratch — then overwrites the
    # finished LoRA with a no-op. Refuse that outright.
    _incompat = network.load_state_dict(load_file(os.path.join(state_dir, "lora.safetensors")), strict=False)
    _missing = getattr(_incompat, "missing_keys", [])
    if _missing and len(_missing) >= len(network.state_dict()):
        raise RuntimeError(
            f"[state] {state_dir} matched none of this network's {len(network.state_dict())} keys — "
            f"refusing to resume into a zero-initialised network. The state was almost certainly "
            f"saved with a different config (rank/alpha/factor, network type, or target modules).")
    opt_path = os.path.join(state_dir, "optimizer.pt")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    rng_path = os.path.join(state_dir, "rng.pt")
    if os.path.exists(rng_path):
        try:
            rng = torch.load(rng_path)
            torch.set_rng_state(rng["torch"].to("cpu", dtype=torch.uint8) if hasattr(rng["torch"], "to") else rng["torch"])
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            logger.warning("[state] RNG restore failed; continuing with fresh RNG", exc_info=True)
    meta_path = os.path.join(state_dir, "training_state.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    return int(meta.get("epoch", 0)), int(meta.get("global_step", 0)), meta


def _save_lora(network, path, network_dim, network_alpha, dtype, extra_metadata=None):
    is_lokr = getattr(network, "_network_type", "lora") == "lokr"
    if is_lokr:
        metadata = {
            "ss_network_module": "fizgig.minimax (lokr, transformer blocks)",
            "ss_lokr_factor": str(getattr(network, "_lokr_factor", "")),
            "ss_architecture": ARCHITECTURE_MINIMAX,
        }
    else:
        metadata = {
            "ss_network_module": "fizgig.minimax (lora_unet, transformer blocks)",
            "ss_network_dim": str(network_dim),
            "ss_network_alpha": str(network_alpha),
            "ss_architecture": ARCHITECTURE_MINIMAX,
        }
    if extra_metadata:
        metadata.update(extra_metadata)
    if is_lokr:
        # LyCORIS-standard keys (diffusion_model.<dotted>.lokr_*) — the format every ComfyUI
        # LoKR in the wild uses. Unlike Krea 2 (whose internal saves stay native for resume and
        # previews), MiniMax has neither, and every checkpoint's only consumer is ComfyUI — so
        # every LoKR save is comfy-format. Fizgig's own loader ingests both namings via
        # ensure_kohya_lora_state_dict.
        from fizgig.networks.lora import _precalculate_safetensors_hashes
        from safetensors.torch import save_file
        dotted = getattr(network, "_dotted_names", {})
        sd = {}
        for k, v in network.state_dict().items():
            mod, _, suffix = k.partition(".")
            path_dotted = dotted.get(mod)
            nk = f"diffusion_model.{path_dotted}.{suffix}" if path_dotted else k
            v = v.detach().clone().to("cpu")
            if dtype is not None:
                v = v.to(dtype)
            sd[nk] = v
        model_hash, legacy_hash = _precalculate_safetensors_hashes(sd, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash
        save_file(sd, path, metadata)
        return
    network.save_weights(path, dtype, metadata)


def train_minimax(
    dataset_config: str,
    output_dir: str,
    output_name: str,
    dit_path: str,
    *,
    network_dim: int = 16,
    network_alpha: float = 16,
    network_type: str = "lora",      # "lora" | "lokr" (Kronecker, full-matrix w2)
    lokr_factor: int = 8,            # LoKR only: w1 is ~factor x factor; dim/alpha unused
    learning_rate: float = 1e-4,
    max_train_epochs: int = 10,
    save_every_n_epochs: int = 0,
    # Resumable state dirs (network + optimizer + RNG + adaptive scalars). Pause saves state
    # regardless of these — they govern only the automatic per-checkpoint / end-of-run saves.
    save_state: bool = False,
    save_state_on_train_end: bool = False,
    keep_last_n_states: int = 2,
    resume_state_dir: str = None,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    optimizer_type: str = "adamw8bit",
    optimizer_args: str = "",
    caption_dropout: float = 0.05,
    base_quant: str = "auto",
    include_patterns: list = None,
    train_blocks: str = None,        # "14-37" = train only that block range (experiment)
    train_adaln: bool = True,        # False = drop adaln_proj from the targets (pruned only)
    distill: bool = False,           # reference distillation (references come from the dataset)
    distill_weight: float = 0.8,     # teacher share of the loss; the rest is the real photo
    distill_phase1_epochs: int = -1,  # identity-first: teacher-ONLY epochs, then photos-only
                                      # (-1 = auto from dataset size, 0 = off/blended)
    slow_blocks: str = None,         # block spec trained at a reduced LR ("21-49")
    block_limit: float = 0.0,   # >0 = per-block movement cap at N x the median block (the limiter)
    adapter_ramp: float = 0.0,  # >0 = hold each step at this FRACTION of the adapter's size
    gradient_accumulation_steps: int = 1,  # batches summed per optimizer step (effective batch)
    lr_warmup_epochs: float = 0.0,  # >0 = linear LR ramp over the first N epochs (static LR only)
    ema_decay: float = 0.0,     # >0 = save/preview the EMA of the adapter instead of raw weights
    slow_block_lr_scale: float = 1.0,  # the multiplier applied to those blocks' LR
    quantize: bool = True,           # NF4 the base (QLoRA); False = bf16 base (needs ~66 GB VRAM)
    shift: float = None,             # None = auto resolution schedule (logit-normal); float = legacy
    blocks_to_swap="auto",           # "auto" | int — park the last N blocks on CPU between uses
    gradient_checkpointing="auto",   # "auto" | "on" | "off" — forced on when swap > 0
    adaptive_lr: bool = False,
    adaptive_lr_min: float = 1e-5,
    adaptive_lr_max: float = 4e-4,
    # In-training previews. Prompts come from the Samples tab; the text encoder is loaded ONCE
    # before the DiT (it must never be resident alongside it) and freed.
    sample_prompts: list = None,
    te_path: str = None,
    vae_path: str = None,
    sample_every_n_epochs: int = 0,
    sample_at_first: bool = False,
    # H3's native canvas: 768 short edge, 768*1344 pixel cap.
    sample_width: int = 768,
    sample_height: int = 768,
    # 28, matching the reference pipeline's default. 8 leaves the latent well off the
    # encoder's manifold, which is exactly where the decoder produces patchy output
    # (measured seam energy 4.0 on an off-manifold latent vs 1.05 on a real one).
    sample_steps: int = 28,
    sample_cfg_scale: float = 1.0,
    sample_frames: int = 1,      # pixel frames on the 17n+5 grid; 1 = classic still
    sample_negative: str = None,
    sample_seed: int = 42,
    # Output metadata (recorded in the saved LoRA).
    metadata_title: str = None,
    metadata_author: str = None,
    metadata_description: str = None,
    metadata_license: str = None,
    metadata_tags: str = None,
    metadata_trigger_phrase: str = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Native MiniMax H3 image-only LoRA training: bucketed dataloader over the H3 caches ->
    flow-matching loss -> optimizer -> save a ComfyUI-compatible LoRA. No samples, no preview."""
    from torch.utils.data import DataLoader

    from fizgig.dataset.config import (BlueprintGenerator, ConfigSanitizer,
                                       generate_dataset_group_by_blueprint, load_user_config)
    from fizgig.networks.lora import create_network
    from fizgig.training.optimizers import create_optimizer
    from fizgig.training.train_utils import LossRecorder
    from fizgig.training.metadata import build_metadata, resolve_title, ARCHITECTURE_MINIMAX
    from fizgig.minimax.loader import load_minimax_h3_dit
    from tqdm import tqdm
    import math

    torch.manual_seed(seed)
    user_include_patterns = include_patterns   # None -> resolved per checkpoint below
    # Parse the block selection NOW, before the 21 GB base streams in: a typo surfacing after
    # the load costs minutes and reads like a crash rather than a correction. Bounds-checking
    # waits until the model is up (that is when the real block count is known).
    if train_blocks:
        parse_block_spec(train_blocks)

    # ---- dataset (built from the caches the two cache scripts wrote) ----
    shared_epoch = Value("i", 0)
    user_config = load_user_config(dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config, argparse.Namespace(), architecture=ARCHITECTURE_MINIMAX)
    group = generate_dataset_group_by_blueprint(
        blueprint.dataset_group, training=True, num_timestep_buckets=None, shared_epoch=shared_epoch)
    if group.num_train_items == 0:
        raise RuntimeError("No training items — run minimax_cache_latents then minimax_cache_text first.")
    logger.info(f"MiniMax H3 training: {group.num_train_items} items, {max_train_epochs} epochs")
    if shift is None:
        logger.info("[timesteps] shift-12 uniform map (median sigma ~0.92) — H3's own training "
                    "density, matching the reference trainer")
    elif shift == "sigmoid":
        logger.warning("[timesteps] UNSHIFTED logit-normal (median 0.5) — A/B mode. At 1e-4 this "
                       "overdrives adapters within an epoch on small datasets; the default "
                       "(omit --shift) is the reference recipe.")
    elif shift == "resolution":
        logger.warning("[timesteps] logit-normal + resolution shift (median ~0.62) — A/B mode; "
                       "same overdrive caveat as sigmoid.")
    elif isinstance(shift, str) and str(shift).startswith("lognorm:"):
        logger.info(f"[timesteps] logit-normal base at shift {str(shift).split(':', 1)[1]} — "
                    f"mid-concentrated spread at the requested low-noise share.")
    else:
        logger.info(f"[timesteps] explicit shift={shift} — uniform-u map.")

    # ---- VRAM plan: block swap + gradient checkpointing (before the base loads) ----
    _mp = 0.25
    try:
        _mp = max(w * h / 1e6 for ds in group.datasets for (w, h) in ds.batch_manager.bucket_resos)
    except Exception:
        pass
    _ckpt_req = str(gradient_checkpointing).lower()
    _base_mode = (base_quant if base_quant != "auto"
                  else ("int8" if is_pruned_checkpoint(dit_path) else "nf4"))
    if not quantize:
        _base_mode = "none"
    if str(blocks_to_swap).lower() == "auto":
        if torch.cuda.is_available() and quantize:
            _free_gb = torch.cuda.mem_get_info()[0] / 1e9
            _pruned = is_pruned_checkpoint(dit_path)
            # The adapter is NOT a rounding error and it is not fixed: LoKR 8 trains ~313 M
            # parameters against a rank-16 LoRA's ~75 M, and fp32 Adam state is 4x the 8-bit
            # one. Planning without it was planning for a configuration nobody runs — the
            # anchors were measured on rank-16 + adamw8bit (~0.45 GB) while the shipped default
            # is LoKR 8 + adamw (~3.8 GB). Shapes come from the checkpoint header, so this is
            # the real targeted module set for whichever file is loaded.
            _pat = PRUNED_INCLUDE_PATTERNS if _pruned else DEFAULT_INCLUDE_PATTERNS
            if not train_adaln:
                _pat = [p for p in _pat if "adaln" not in p]
            _ad_params = adapter_param_count(dit_path, _pat, network_type=network_type,
                                             network_dim=network_dim, lokr_factor=lokr_factor,
                                             train_blocks=train_blocks)
            _adapter = adapter_vram_gb(_ad_params, optimizer_type)

            if base_quant == "auto":
                _mode, n_swap, _ckpt_auto, _why = plan_base_quant(
                    _free_gb, _pruned, mp=_mp, adapter_gb=_adapter)
            else:
                # An explicit choice is never overridden — the plan is built AROUND it, or the
                # swap count would be sized for a quantisation that will not run.
                _mode = base_quant
                _res = (_RESIDENT_INT8_GB if _mode == "int8"
                        else _RESIDENT_PRUNED_GB if _pruned else _RESIDENT_GB)
                n_swap, _ckpt_auto = plan_vram(
                    _free_gb, mp=_mp, resident_gb=_res,
                    transient_gb=_INT8_TRANSIENT_GB if _mode == "int8" else 0.0,
                    adapter_gb=_adapter)
                _why = f"base precision pinned to {_mode} by the user"
            _base_mode = _mode
            _resident = (_RESIDENT_INT8_GB if _mode == "int8"
                         else _RESIDENT_PRUNED_GB if _pruned else _RESIDENT_GB)

            logger.info(f"[vram] auto plan: free {_free_gb:.1f} GB, largest bucket {_mp:.2f} MP, "
                        f"base ~{_resident:.0f} GB ({_mode}, {'pruned' if _pruned else 'bf16'}), "
                        f"adapter ~{_adapter:.1f} GB ({_ad_params/1e6:.0f} M params, "
                        f"{optimizer_type}) -> blocks_to_swap={n_swap}, "
                        f"checkpointing={'on' if _ckpt_auto else 'off'}")
            logger.info(f"[vram] base precision: {_mode} — {_why}")
            if _mode == "nf4" and _pruned and base_quant == "auto":
                # Say it plainly rather than quietly downgrading the base: this costs likeness,
                # and the user has a real alternative (train slower on int8, or free some VRAM).
                logger.warning(
                    "[vram] this run trains on a 4-bit base (~9% error) instead of the "
                    "checkpoint's own int8 (~0.17%). It is much faster here, but the LoRA spends "
                    "some capacity correcting quantization error that will NOT exist at "
                    "inference. To force the accurate base, set Base Precision to int8 — expect "
                    "block swap and a several-times-slower run — or close other GPU apps and "
                    "re-launch.")
            if n_swap > 0:
                logger.warning(
                    f"[vram] {n_swap} of 50 blocks will live on CPU and cross PCIe every step, "
                    f"which is several times slower. Lower Target Megapixels, or free VRAM, to "
                    f"avoid it.")
        else:
            n_swap, _ckpt_auto = 0, False
    else:
        n_swap = max(0, int(blocks_to_swap))
        _ckpt_auto = n_swap > 0
        if base_quant == "auto" and quantize:
            # A hand-set swap count skips the planner, and the planner is what weighs int8's
            # residency against free VRAM. Auto precision then falls back to "whatever the file
            # is", which on a pruned checkpoint is always int8 — ~21 GB and a slow run on a card
            # the planner would have put on 4-bit with no swap at all. It does not fail, so
            # nothing would otherwise say why the run is crawling.
            logger.info(f"[vram] base precision: {_base_mode} — chosen from the checkpoint, not "
                        f"from free VRAM, because Blocks Swap is set to {n_swap} rather than "
                        f"Auto. Set Blocks Swap to Auto to have the precision and the swap count "
                        f"planned together.")
    use_ckpt = {"on": True, "off": False}.get(_ckpt_req, _ckpt_auto)
    if n_swap > 0 and not use_ckpt:
        logger.info("[vram] block swap needs gradient checkpointing (autograd would pin swapped "
                    "weights through backward) — forcing it on.")
        use_ckpt = True

    # ---- previews: encode the prompts BEFORE the DiT loads ----
    # Order matters more here than anywhere else in Fizgig: the Qwen3-VL-32B text encoder is
    # ~14 GB even at NF4, and the DiT is ~17 GB. They must never be resident together, so the
    # prompts are encoded once, up front, and the encoder is freed before the base streams in.
    do_previews = bool((sample_every_n_epochs or sample_at_first) and sample_prompts and te_path)
    encoded_prompts = encoded_negative = sample_dir = None
    if do_previews:
        from fizgig.minimax.sampling import encode_sample_prompts
        logger.info(f"[preview] pre-encoding {len(sample_prompts)} sample prompt(s) "
                    f"(the text encoder is freed before the DiT loads)...")
        try:
            encoded_prompts = encode_sample_prompts(te_path, sample_prompts, device=device,
                                                    quantize=quantize)
            if sample_negative and sample_cfg_scale and sample_cfg_scale > 1.0:
                encoded_negative = encode_sample_prompts(te_path, [sample_negative],
                                                         device=device, quantize=quantize)[0]
            sample_dir = os.path.join(output_dir, "sample")
            os.makedirs(sample_dir, exist_ok=True)
            # State the whole preview recipe once, up front — steps in particular, since too
            # few leaves the latent off-manifold and the decode patchy.
            logger.info(
                f"[preview] {sample_steps} steps @ {sample_width}x{sample_height}, "
                f"cfg {sample_cfg_scale:g}"
                f"{'' if sample_cfg_scale > 1.0 else ' (off — H3 is guidance-distilled)'}, "
                f"seed {sample_seed if sample_seed else 'random'}, "
                f"every {sample_every_n_epochs} epoch(s)"
                f"{', plus epoch 0' if sample_at_first else ''} — "
                f"{'full VAE decode' if vae_path else 'RGB approximation (no VAE path set)'}")
        except Exception as _e:
            logger.warning(f"[preview] prompt encoding failed ({type(_e).__name__}: {_e}) — "
                           f"previews disabled; training continues normally.")
            do_previews = False

    # ---- base (NF4-frozen) + trainable LoRA over the transformer blocks ----
    # adaln_fp32 matches ComfyUI's curve-checkpoint dtype, but only when AdaLN is NOT a LoRA
    # target — a bf16 adapter cannot take an fp32 activation from the Linear it wraps.
    # base_quant is the RESOLVED mode, never the raw "auto" (issue #55). The loader has its own
    # auto rule — int8 whenever the file is pre-quantized — which ignores how much VRAM is free,
    # so handing it "auto" threw away the plan: swap sized for an ~11 GB NF4 base while a ~21 GB
    # int8 one loaded, and the log printed both decisions a few lines apart. It also made
    # ss_base_quant a lie in the output metadata.
    dit = load_minimax_h3_dit(dit_path, device=device, compute_dtype=dtype, quantize=quantize,
                              blocks_to_swap=n_swap, base_quant=_base_mode,
                              adaln_fp32=not train_adaln)
    dit.requires_grad_(False)                                   # frozen base (QLoRA-style)
    if n_swap > 0:
        n_swap = dit.enable_block_swap(n_swap)                  # sets the JIT-move boundary
        logger.info(f"[vram] block swap active: last {n_swap} blocks parked on CPU "
                    f"(~{n_swap * 0.34:.1f} GB VRAM freed, packed NF4 in RAM)")
    if use_ckpt:
        dit.enable_gradient_checkpointing()
        logger.info("[vram] gradient checkpointing ON")
    # AdaLN targeting is per-checkpoint — see the pattern note at the top of this file.
    include_patterns = user_include_patterns or (
        PRUNED_INCLUDE_PATTERNS if dit.pruned_adaln else DEFAULT_INCLUDE_PATTERNS)
    # AdaLN is a pure function of the TIMESTEP — DiTBlock.forward calls adaln_proj(t_emb) and
    # nothing else, so its adapters cannot tell one subject from another. They can only reshape
    # how strongly each block fires at each noise level. On the pruned checkpoint they carry
    # ~45% of all weight movement in a matched epoch, which is a lot of a LoRA's capacity spent
    # somewhere structurally incapable of holding a face — hence the toggle. See
    # docs/MINIMAX_BLOCKS.md. No-op on the bf16 checkpoint, which never targets AdaLN.
    _adaln_on = bool(train_adaln) and dit.pruned_adaln
    if not train_adaln:
        _before = len(include_patterns)
        include_patterns = [p for p in include_patterns if "adaln" not in p]
        if len(include_patterns) < _before:
            logger.info("[base] EXPERIMENT: AdaLN adapters OFF. AdaLN sees only the timestep, so "
                        "it cannot encode identity — this frees the capacity it was taking. "
                        "Compare against the same run with it on.")
        else:
            logger.info("[base] AdaLN was not a target on this checkpoint; the toggle changes "
                        "nothing here.")
    _blocks_used = "all"
    if train_blocks:
        _n_blocks = len(dit.blocks)
        include_patterns = restrict_patterns_to_blocks(include_patterns, train_blocks, _n_blocks)
        _sel = parse_block_spec(train_blocks, _n_blocks)
        _blocks_used = format_block_spec(_sel)
        logger.info("[base] EXPERIMENT: training blocks %s only (%d of %d), text refiner "
                    "included. Nobody has mapped what H3's blocks do — judge this against a "
                    "full-model run on the same dataset, not on its own.",
                    _blocks_used, len(_sel), _n_blocks)
    # Report what is ACTUALLY targeted: this used to key off the checkpoint alone, so a run with
    # --no_train_adaln announced "+ AdaLN" one line after saying AdaLN adapters were off.
    _adaln_on = bool(dit.pruned_adaln and train_adaln)
    logger.info("[base] %s checkpoint; LoRA targets: attention + MLP + token refiner%s",
                "pruned (curve-table AdaLN)" if dit.pruned_adaln else "full bf16",
                " + AdaLN (deploy-consistent on this build; rank caps at 8)" if _adaln_on
                else (" (AdaLN excluded - turned off for this run)" if dit.pruned_adaln
                      else " (AdaLN excluded - dropped by pruned inference builds)"))
    if network_type == "lokr":
        # LoKR (Kronecker) — same mechanism as Krea 2's: module_class swaps the parametrization
        # inside the identical scan/wrap machinery, so include_patterns (adaln exclusion) and the
        # NF4/Linear4bit base compose unchanged. dim/alpha are ignored; factor is the dial.
        from fizgig.networks.lora import LoKRModule
        logger.info(f"network: LoKR (Kronecker), factor {lokr_factor}, full-matrix w2 — "
                    "dim/alpha do not apply")
        network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit,
                                 include_patterns=include_patterns,
                                 module_class=LoKRModule, module_kwargs={"factor": int(lokr_factor)})
    else:
        network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit,
                                 include_patterns=include_patterns)
    network.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    network.to(device=device, dtype=dtype)
    network._network_type = network_type
    network._lokr_factor = int(lokr_factor)
    # Dotted module paths for the LyCORIS-standard save (diffusion_model.<path>.lokr_*) — built
    # from the DiT itself with the same flattening create_modules used, so the reverse mapping is
    # exact even where module names contain underscores. isinstance covers bnb Linear4bit (an
    # nn.Linear subclass).
    network._dotted_names = {
        f"lora_unet_{name.replace('.', '_')}": name
        for name, m in dit.named_modules() if isinstance(m, torch.nn.Linear)
    }
    _n_targeted = len(network.unet_loras)
    if network_type == "lokr":
        logger.info(f"LoKR: {len(network.unet_loras)} modules wrapped (factor {lokr_factor})")
    else:
        logger.info(f"LoRA: {len(network.unet_loras)} modules wrapped (dim {network_dim}, alpha {network_alpha})")

    # How many Linears did the include_patterns actually TARGET? create_modules matches by class
    # NAME, so a quantized Linear stand-in that is not on that list is skipped in silence — which
    # once shipped a run training 58 of 258 modules with no error anywhere. Compare and refuse.
    import re as _re
    _targeted = [n for n, m in dit.named_modules()
                 if isinstance(m, torch.nn.Linear)
                 and any(_re.search(p, n) for p in include_patterns)]
    if len(network.unet_loras) < len(_targeted):
        _kinds = sorted({type(dit.get_submodule(n)).__name__ for n in _targeted})
        raise RuntimeError(
            f"only {len(network.unet_loras)} of {len(_targeted)} targeted Linears were wrapped — "
            f"the network builder matches by class name and one of {_kinds} is not on its list "
            f"(networks/lora.py, create_modules). Training now would silently learn a fraction "
            f"of the model.")
    _n_targeted = len(_targeted)
    logger.info(f"[network] {len(network.unet_loras)}/{_n_targeted} targeted Linears wrapped")

    params = list(network.get_trainable_params())

    # Adaptive LR ignores the Learning Rate box: it starts at the GEOMETRIC MIDPOINT of Min/Max
    # and the watcher owns the LR from there (matches Klein/Krea 2). Two knobs, not three.
    adaptive = AdaptiveLR(adaptive_lr_min, adaptive_lr_max) if adaptive_lr else None
    if adaptive:
        learning_rate = math.sqrt(adaptive_lr_min * adaptive_lr_max)
        logger.info(f"[adaptive_lr] ENABLED — start_lr={learning_rate:.3e} (geometric midpoint) "
                    f"min={adaptive_lr_min:.3e} max={adaptive_lr_max:.3e}; the Learning Rate box is ignored")

    # Weight-decay parity with the reference trainer: ai-toolkit's job template passes
    # optimizer_params weight_decay=1e-4; bitsandbytes' default is 0.01 (100x). Only applied
    # when the user hasn't set their own via Optimizer Args.
    if "weight_decay" not in (optimizer_args or "") and "adam" in optimizer_type.lower():
        optimizer_args = (optimizer_args + " weight_decay=1e-4").strip()

    # Depth-dependent LR. A perturbation injected at block 5 passes through 45 more blocks that
    # absorb and renormalize it; one injected at block 45 lands almost directly on the output. So
    # the same |dW| is far more disruptive the later it sits, and ONE learning rate is wrong by
    # construction — it is either too low for the early blocks or too high for the late ones.
    # Observed here: at 1e-4, blocks 0-20 train cleanly but slowly while anything past 20 wrecks
    # the samples (block swap ruled out — those runs recorded blocks_swapped=0).
    # Built AFTER the adaptive block above, so `learning_rate` is already the resolved start LR.
    _slow_used, _slow_n = "", 0
    opt_params = params          # the optimizer may get groups; `params` stays flat for clipping
    if slow_blocks and abs(float(slow_block_lr_scale) - 1.0) > 1e-9:
        _slow_idx = set(parse_block_spec(slow_blocks, len(dit.blocks)))
        _slow_ids = set()
        for _lora in network.unet_loras:
            _nm = _lora.lora_name
            if "token_refiner" in _nm:      # text-side, never part of the depth argument
                continue
            _m = re.search(r"blocks_(\d+)_", _nm)
            if _m and int(_m.group(1)) in _slow_idx:
                _slow_ids.update(id(p) for p in _lora.parameters())
        if _slow_ids:
            _slow = [p for p in params if id(p) in _slow_ids]
            _fast = [p for p in params if id(p) not in _slow_ids]
            _scaled = learning_rate * float(slow_block_lr_scale)
            # lr_scale rides along on the group so the adaptive watcher can move both groups
            # together without flattening them back to one rate.
            # NOTE: assign to opt_params, NOT params. `params` stays the flat tensor list because
            # clip_grad_norm_ iterates it every step and cannot take param-group dicts.
            opt_params = [{"params": _fast, "lr": learning_rate, "lr_scale": 1.0},
                          {"params": _slow, "lr": _scaled, "lr_scale": float(slow_block_lr_scale)}]
            _slow_used = format_block_spec(sorted(_slow_idx))
            _slow_n = len(_slow)
            logger.info("[lr] depth-split: blocks %s train at %.3e (x%g), the rest at %.3e "
                        "(%d of %d tensors slowed)", _slow_used, _scaled, slow_block_lr_scale,
                        learning_rate, _slow_n, len(_slow) + len(_fast))
        else:
            logger.warning("[lr] slow_blocks %r matched no trained modules — is it outside "
                           "Blocks to Train? Depth-split LR is not active.", slow_blocks)

    # eps_floor_8bit: H3-only. The 8-bit second moment underflows on this model's most structured
    # tensors and the update degrades to lr*m/eps — measured at ~100x the configured LR, which
    # presented as melted anatomy at epoch 1. The floor caps that. It is passed here and nowhere
    # else: Krea 2 has never shown the failure and keeps the library default.
    optimizer, optimizer_label = create_optimizer(optimizer_type, opt_params, learning_rate,
                                                  optimizer_args, eps_floor_8bit=True)
    logger.info(f"optimizer: {optimizer_label} @ lr={learning_rate:.3e}")

    limiter = None
    if block_limit and float(block_limit) > 0:
        limiter = StepClipper(network, float(block_limit))
        logger.info(f"[clip] per-step movement cap ON at {float(block_limit):g}x the median "
                    f"block ({len(limiter.groups)} blocks watched) — whichever block overshoots "
                    f"in a single step gets pulled back to the pack, wherever the trained range "
                    f"ends. Only the offending STEP is shortened; nothing already learned is "
                    f"scaled down.")

    ramp = None
    if adapter_ramp and float(adapter_ramp) > 0:
        if adaptive is not None:
            logger.info("[ramp] ignored — Adaptive LR owns the schedule.")
        else:
            ramp = AdapterRamp(network, float(adapter_ramp))
            logger.info(f"[ramp] adapter-relative LR ON — each step held at "
                        f"{100 * float(adapter_ramp):.3f}% of the adapter's current size, so the "
                        f"LR starts low and climbs toward your configured ceiling as the adapter "
                        f"grows. A step is a huge perturbation of a new adapter and a small one "
                        f"of a mature adapter; this keeps the RATIO steady instead of the rate.")

    ema = None
    if ema_decay and float(ema_decay) > 0:
        ema = EMAWeights(network, float(ema_decay))
        logger.info(f"[ema] ON at decay {float(ema_decay):g} — checkpoints and previews use the "
                    f"smoothed average of the training path; training itself runs on the raw "
                    f"weights. Big-LR strides zigzag; the EMA is the center of the zigzag.")

    # LR warmup: the epoch-1 damage from a high static LR comes from oversized strides landing
    # on zero-init adapters at the steepest part of the loss surface. A linear ramp over the
    # first N epochs eases in, then runs at full speed — the cost is a fraction of one epoch's
    # worth of movement. Adaptive LR owns its own schedule, so warmup only applies without it.
    if adaptive is not None and lr_warmup_epochs and lr_warmup_epochs > 0:
        logger.info("[warmup] ignored — Adaptive LR owns the schedule (it already starts at "
                    "the midpoint and probes from there).")
        lr_warmup_epochs = 0.0

    # Caption dropout (reference default 0.05): swap in the cached empty-prompt embed for a
    # random ~5% of steps. The uncond file is written by minimax_cache_text next to the caches.
    uncond_text = None
    if caption_dropout and caption_dropout > 0:
        for _ds in group.datasets:
            _f = os.path.join(getattr(_ds, "cache_directory", "") or "",
                              f"uncond_{ARCHITECTURE_MINIMAX}_te.safetensors")
            if os.path.isfile(_f):
                from safetensors.torch import load_file as _lf
                uncond_text = _lf(_f)["hidden_states"].unsqueeze(0)      # (1, L, 5120)
                break
        if uncond_text is None:
            logger.warning("[caption_dropout] no uncond embed in the cache dirs (re-run text "
                           "caching to enable it) — dropout disabled for this run")
        else:
            logger.info(f"[caption_dropout] {caption_dropout:.2f} — empty-prompt embed loaded")

    # Reference distillation needs nothing at run start: each item's reference conditioning AND
    # that reference's latent both ride in from the cache, one slot picked at random per step.
    if distill:
        logger.info("[distill] reference distillation ON — teacher weight %.2f, photo %.2f. "
                    "References come from the dataset itself (each image paired with others by "
                    "the caching pass); no image is ever its own reference.",
                    distill_weight, 1.0 - distill_weight)

    collator = _Collator(shared_epoch, group)
    loader = DataLoader(group, batch_size=1, shuffle=True, collate_fn=collator, num_workers=0)
    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = group.num_train_items

    # --- identity-first (two-phase distillation) --------------------------------------------
    # Phase 1 trains ONLY against the teacher, so the adapter learns who the trigger means
    # before it is asked to reproduce any particular photograph. Phase 2 then drops the teacher
    # entirely and trains on the photos alone, starting from an adapter that already has the
    # identity in the right places. A hard switch, not a decay: this is an INITIALISATION
    # strategy, so what phase 2 forgets about the teacher does not matter.
    #
    # AUTO length comes from a real run (11 Aug, 82 images): the teacher error fell 0.069 ->
    # 0.051 -> 0.050 over epochs 7-9 — converged by epoch 8, i.e. ~650 gradient STEPS. Steps,
    # not epochs, is the invariant: a 24-image set needs the same number of steps, which is
    # many more epochs. Held at one epoch minimum.
    _p1_epochs = 0
    if distill:
        _p1_epochs = (max(1, math.ceil(650 / max(1, steps_per_epoch)))
                      if distill_phase1_epochs is None or distill_phase1_epochs < 0
                      else int(distill_phase1_epochs))
        _p1_epochs = min(_p1_epochs, max_train_epochs)
        if _p1_epochs > 0:
            logger.info(
                f"[distill] IDENTITY-FIRST: epochs 1-{_p1_epochs} train against the teacher "
                f"ONLY (~{_p1_epochs * steps_per_epoch} steps) at "
                f"{learning_rate * _P1_LR_SCALE:.2e} — a third of the box — then the teacher is "
                f"dropped and epochs {_p1_epochs + 1}-{max_train_epochs} train on the "
                f"photographs alone at the full {learning_rate:.2e}. "
                f"The teacher weight box does not apply in this mode."
                + ("" if distill_phase1_epochs is not None and distill_phase1_epochs >= 0 else
                   "  (length chosen from the dataset size — the teacher objective converges in "
                   "roughly 650 steps whatever the image count.)"))

    # Gradient accumulation. Batch size 1 means every step is aimed by ONE image, so a large
    # stride follows an equally large random walk — the roughness that reads as "quality loss
    # without distortion" when a run covers ground fast. Averaging the gradient over N images
    # before stepping makes a big step PRECISE instead of rough, at the same wall-clock per
    # epoch (same forwards, N times fewer optimizer steps).
    _accum_n = max(1, int(gradient_accumulation_steps or 1))
    if _accum_n > 1:
        logger.info(f"[accum] gradient accumulation {_accum_n} — effective batch {_accum_n}, "
                    f"{steps_per_epoch // _accum_n} optimizer steps per epoch instead of "
                    f"{steps_per_epoch}. Each step is aimed by {_accum_n} images, so the same "
                    f"stride carries far less sampling noise.")

    warmup_steps = int(round(float(lr_warmup_epochs or 0.0) * steps_per_epoch))
    if warmup_steps > 0:
        logger.info(f"[warmup] LR ramps linearly over the first {lr_warmup_epochs:g} epoch(s) "
                    f"= {warmup_steps} steps, then holds at the configured LR.")

    os.makedirs(output_dir, exist_ok=True)
    pause_flag = os.path.join(output_dir, ".pause_requested")

    # ---- resume: restore network + optimizer + RNG + (epoch, step) + adaptive scalars ----
    from fizgig.training.train_utils import prune_state_dirs
    global_step = 0
    start_epoch = 0
    # `if resume_state_dir` — NOT `and os.path.isdir(...)`: a requested resume whose path is bad
    # used to skip this block silently and train from scratch. Resume, or refuse — never ignore.
    if resume_state_dir:
        start_epoch, global_step, _resume_meta = _load_training_state(
            resume_state_dir, network, optimizer, device=device)
        if adaptive:
            adaptive.load_state_dict(_resume_meta.get("adaptive_lr_state"))
        if ema is not None:
            _ema_path = os.path.join(resume_state_dir, "ema.pt")
            if os.path.isfile(_ema_path):
                ema.load_state_dict(torch.load(_ema_path, map_location="cpu"))
                logger.info(f"[ema] restored the running average ({ema.n} updates)")
            else:
                # The state predates EMA (or it was off then). Restart the average from the
                # RESTORED weights — the shadow currently holds the zero init from construction.
                ema.shadow = [p.detach().clone().float() for p in ema.params]
                logger.info("[ema] no EMA state in the resume dir — restarting the average "
                            "from the restored weights.")
        _rs = _resume_meta.get("adapter_ramp")
        if ramp is not None and _rs:
            ramp.mult = float(_rs.get("mult", ramp.mult))
            ramp._smooth = (float(_rs["smooth"]) if _rs.get("smooth") is not None else None)
            ramp._prev = (float(_rs["prev"]) if _rs.get("prev") is not None else None)
            logger.info(f"[ramp] restored — LR resumes at {100 * ramp.mult:.0f}% of the "
                        f"configured ceiling rather than re-climbing from the floor")
        logger.info(f"[resume] from {resume_state_dir}: continuing at epoch "
                    f"{start_epoch + 1}/{max_train_epochs} (global_step {global_step})")
        if start_epoch >= max_train_epochs:
            # Pausing ON the last epoch exits before the final LoRA is written — Resume is what
            # completes it, so this fall-through writes the final file from the restored state.
            logger.warning(f"[resume] state is at epoch {start_epoch} of {max_train_epochs} — "
                           f"nothing left to train. Writing the final LoRA from the restored "
                           f"state. To train further, raise Max Train Epochs and resume again.")

    if warmup_steps > 0 or ramp is not None or _p1_epochs:
        # Stashed AFTER the resume block: optimizer.load_state_dict replaces the param-group
        # dicts, so a stash made earlier would not survive a resume. Derived from the CONFIGURED
        # rate (x the group's depth-split scale), not the group's current lr, which a resumed
        # mid-ramp state would have left partway up.
        for _g in optimizer.param_groups:
            _g["_warmup_base_lr"] = learning_rate * float(_g.get("lr_scale", 1.0))
    # WHOEVER OWNS THE LR SETS IT — and when nobody does, the configured value must win.
    # NOT an elif on the block above: warmup CONFIGURED but already FINISHED lands here too,
    # and that was exactly the case the first version of this fix missed.
    if should_reassert_lr(resuming=bool(resume_state_dir), adaptive=adaptive, ramp=ramp,
                          warmup_steps=warmup_steps, global_step=global_step):
        _stale = float(optimizer.param_groups[0].get("lr", learning_rate))
        for _g in optimizer.param_groups:
            _g["lr"] = learning_rate * float(_g.get("lr_scale", 1.0))
        if abs(_stale - learning_rate) > 1e-12:
            logger.info("[resume] the saved state carried lr=%.3e (a throttled value from when "
                        "it was written); nothing is modulating the LR this run, so the "
                        "configured %.3e is reasserted.", _stale, learning_rate)

    def _run_provenance():
        """What actually produced this LoRA — the facts you need to compare two of them.

        Added after an A/B where the file could not answer "was this the int8 base or NF4?",
        "how many modules were really wrapped?" or "how many steps?" — all of which changed the
        interpretation completely, and one of which (58 of 258 modules) had been a silent bug.
        A LoRA that cannot describe its own run is a measurement you have to take on trust."""
        try:
            _res = sorted({f"{w}x{h}" for ds in group.datasets
                           for (w, h) in ds.batch_manager.bucket_resos})
        except Exception:
            _res = []
        _dens = ("shift12" if shift is None else
                 shift if isinstance(shift, str) else f"shift{shift:g}")
        return {
            "ss_base_checkpoint": os.path.basename(dit_path),
            "ss_base_quant": _base_mode,
            "ss_lora_modules": str(len(network.unet_loras)),
            "ss_targeted_modules": str(_n_targeted),
            "ss_steps": str(global_step),
            "ss_epochs": str(max_train_epochs),
            "ss_learning_rate": f"{learning_rate:g}",
            "ss_optimizer": optimizer_label,
            "ss_timestep_density": _dens,
            "ss_train_blocks": _blocks_used,
            "ss_train_adaln": "1" if _adaln_on else "0",
            "ss_distill": "dataset" if distill else "off",
            "ss_distill_weight": (f"{distill_weight:g}" if distill else "0"),
            "ss_slow_blocks": _slow_used or "none",
            "ss_block_limit": str(block_limit or 0),
            "ss_gradient_accumulation": str(_accum_n),
            "ss_adapter_ramp": f"{adapter_ramp:g}" if ramp is not None else "0",
            "ss_lr_warmup_epochs": f"{lr_warmup_epochs:g}",
            "ss_ema_decay": f"{ema_decay:g}" if ema is not None else "0",
            "ss_slow_block_lr_scale": (f"{slow_block_lr_scale:g}" if _slow_used else "1"),
            "ss_caption_dropout": f"{caption_dropout:g}" if uncond_text is not None else "0",
            # One [[datasets]] block per subject is how Multi Concept keeps two people apart, so
            # a deployed LoRA should say how many it carries and where they came from — six
            # months later the trigger words are the only other clue.
            "ss_multi_concept": str(len(group.datasets)),
            "ss_concept_dirs": ",".join(
                os.path.basename(str(getattr(d, "image_directory", "") or "").rstrip("/\\"))
                for d in group.datasets),
            "ss_max_grad_norm": f"{max_grad_norm:g}",
            "ss_bucket_resolutions": ",".join(_res),
            "ss_gradient_checkpointing": "1" if use_ckpt else "0",
            "ss_blocks_swapped": str(n_swap),
        }

    def _meta():
        md = build_metadata(
            None, ARCHITECTURE_MINIMAX, time.time(),
            title=(metadata_title if metadata_title is not None
                   else resolve_title(output_name, metadata_trigger_phrase)),
            author=metadata_author, description=metadata_description,
            license=metadata_license, tags=metadata_tags, trigger_phrase=metadata_trigger_phrase)
        md.update(_run_provenance())
        return md

    def _state_extra():
        extra = {}
        if adaptive:
            extra["adaptive_lr_state"] = adaptive.state_dict()
        if ramp is not None:
            # Three JSON-safe scalars. Without them a resume restarts the climb at the floor
            # and spends ~78 steps re-earning a multiplier it had already established — the
            # same defect the retired governor shipped with, so it does not ship again.
            extra["adapter_ramp"] = {"mult": ramp.mult, "smooth": ramp._smooth,
                                     "prev": ramp._prev}
        return extra or None

    # Encoded override prompt, kept between epochs: re-encoding costs a TE load, so only redo it
    # when the prompt text actually changes.
    _ov_state = {"prompt": None, "enc": None}

    def _encode_override(prompt):
        """Encode one override prompt mid-run.

        The TE is ~14.5 GB and the int8 base ~21 GB, so unlike Krea 2 they cannot both be
        resident on a 32 GB card — the normal flow deliberately encodes every prompt BEFORE the
        DiT loads. To honour a live override we park the DiT on CPU for the duration, then
        restore it (and its block-swap split). That is a ~21 GB round trip, which is why the
        result is cached against the prompt text and only paid when you actually change it."""
        from fizgig.minimax.sampling import encode_sample_prompts
        _free = (torch.cuda.mem_get_info()[0] / 1e9) if torch.cuda.is_available() else 0.0
        _park = torch.cuda.is_available() and _free < 17.0     # TE + headroom
        if _park:
            logger.info(f"[sample override] parking the base on CPU to fit the text encoder "
                        f"({_free:.1f} GB free) — one-off for this prompt")
            dit.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
        try:
            return encode_sample_prompts(te_path, [prompt], device=device, quantize=quantize)
        finally:
            if _park:
                dit.to(device)
                if n_swap > 0:
                    dit.enable_block_swap(n_swap)   # restores the parked-block split
                gc.collect()
                torch.cuda.empty_cache()

    # Clip previews carry real failure risk a still never had (a 124-frame clip is ~30x the
    # sampling tokens plus a chunked multi-frame decode), and the epoch loop LATCHES previews
    # off on any preview exception. A clip-specific failure must degrade to a SHORTER clip that
    # fits, not take every future preview down with it — so the frame count lives in mutable
    # state the failure handlers can lower.
    _clip_state = {"frames": max(1, int(sample_frames or 1)), "notice_done": False,
                   "slow_done": False}

    def _slow_step_notice(seconds, step, total):
        """Told once when a preview step runs absurdly long.

        A preview that does not fit in VRAM does NOT raise on Windows — the driver pages to
        system RAM and the render succeeds at roughly a hundred times the cost, so the
        clip->stills fallback (which is exception-driven) never fires and the run looks hung.
        Wall time is the only symptom that survives, so it is what we watch."""
        if _clip_state["slow_done"]:
            return
        _clip_state["slow_done"] = True
        logger.warning(
            f"[preview] step {step}/{total} took {seconds:.0f}s — far slower than this should "
            f"be, which almost always means the preview does not fit in VRAM and is spilling "
            f"into system RAM. It will finish, just slowly. For future previews, lower "
            f"Width/Height on the Samples tab (and Sample length if you are rendering clips). "
            f"Previews are a heartbeat between checkpoints, not the verdict: every epoch saves "
            f"a .safetensors, and you can Pause the run to free the GPU, judge an epoch in "
            f"ComfyUI, then close ComfyUI and Resume.")

    def _render_previews(epoch):
        """Render one still per prompt on the RESIDENT training DiT and write them where the
        samples gallery looks. The filename format is the gallery/likeness/Visualiser contract
        (parse_sample_filename in the GUI) — do not change it casually.

        The DiT never moves: only eval mode is toggled, and block swap's JIT .to() is already
        forward-safe, so there is no swap-mode dance like Krea 2 needs."""
        import time as _time
        import numpy as _np
        from PIL import Image
        from fizgig.minimax import sampling
        was_training = dit.training
        decoder = None
        _base_parked = False        # set in phase 2; the finally MUST restore a parked base
        _opt_parked = []            # set in phase 1; ditto for the parked optimizer state
        _ema_parked = False         # set in phase 1; ditto for the parked fp32 EMA shadow
        try:
            dit.eval()
            if vae_path:
                # Loaded per preview and freed in the finally: the ViT3D decoder is ~4.85 GB and
                # would otherwise sit on top of the resident base for the whole run.
                from safetensors import safe_open as _safe_open
                from fizgig.minimax.vae import MiniMaxH3VideoVAEDecoder
                decoder = MiniMaxH3VideoVAEDecoder()
                with _safe_open(vae_path, framework="pt", device="cpu") as _f:
                    decoder.load_state_dict({k: _f.get_tensor(k) for k in _f.keys()}, strict=False)
                # FP16, not the training dtype and not fp32. ComfyUI allows this VAE exactly
                # [float16, float32] (sd.py:951) where its class default and every neighbouring
                # video VAE also list bfloat16 — bf16 was singled out and removed for this
                # decoder. The weights ship fp16 (minimax_h3_video_vae_fp16.safetensors), so
                # casting to bf16 threw away 3 mantissa bits at load, and 36 pre-norm residual
                # blocks feed a proj_out that emits 3072 pixel values per token: the error lands
                # straight on pixels as softness and gradient banding, with nothing downstream to
                # smooth it. fp16 costs the same 4.8 GB as bf16 (fp32 would be 9.7), so this is
                # free. Overflow is covered by the same nan_to_num guard ComfyUI relies on
                # (vae.py, attention output) — fp16 is the regime that guard was written for.
                # It also stays on CPU until the DECODE phase: previews used to put it on the GPU
                # before sampling even started, which cost the sampling forward 4.85 GB of
                # headroom it never used — harmless for a 256-token still, an OOM for a 124-frame
                # clip whose forward is ~30x the tokens (real 32 GB-card failure, 8 Aug).
                decoder = decoder.to(torch.float16).eval()
            # Live override from the GUI, re-read every epoch so it can be turned on, changed or
            # switched off mid-run without touching the paused/resume path.
            _prompts, _w, _h = encoded_prompts, sample_width, sample_height
            _seed = sample_seed
            _ov = read_sample_override(output_dir)
            if _ov and not te_path:
                logger.warning("[sample override] a prompt is set but no --text_encoder is "
                               "configured, so it cannot be encoded — using the Samples tab.")
                _ov = None
            if _ov:
                if _ov["prompt"] != _ov_state["prompt"]:
                    # A failed encode must not take previews down with it. This loads the
                    # 14.5 GB TE mid-run (parking the 21 GB base to fit), and on a tight card
                    # that can OOM — and an exception from here used to propagate into the
                    # epoch loop's preview catch, which LATCHES previews off for the rest of
                    # the run. One bad encode silently ended every preview and read from the
                    # outside as "the override just stopped working". Fall back to the Samples
                    # tab prompts for this epoch instead; the state is left untouched, so the
                    # next boundary retries.
                    try:
                        _ov_state["enc"] = _encode_override(_ov["prompt"])
                        _ov_state["prompt"] = _ov["prompt"]
                    except Exception as _oe:
                        logger.warning(
                            f"[sample override] could not encode the new prompt "
                            f"({type(_oe).__name__}) — using the Samples tab prompts this "
                            f"epoch; will retry at the next preview.")
                        _ov = None
            if _ov:
                _prompts, _w, _h, _seed = _ov_state["enc"], _ov["width"], _ov["height"], _ov["seed"]
                logger.info(f"[sample override] active — '{_ov['prompt'][:60]}' "
                            f"seed={_seed} {_w}x{_h}")

            _seed = _seed if _seed != 0 else random.randint(1, 2 ** 31 - 1)
            ts = _time.strftime("%Y%m%d%H%M%S")
            _frames = max(1, int(_clip_state["frames"]))
            if _frames > 1 and decoder is None:
                logger.warning("[preview] clip samples need the video VAE for decode — no VAE "
                               "path is configured, so this epoch renders stills instead.")
                _frames = 1
            if _frames > 1 and not _clip_state["notice_done"]:
                _clip_state["notice_done"] = True
                logger.info(f"[preview] clip mode: {_frames} frames per sample at {_w}x{_h} — "
                            f"clips take longer than stills, and longer clips take longer "
                            f"still. Cadence is 'Sample every N epochs' and size is "
                            f"Width/Height, both on the Samples tab.")
            # PHASE 1 — sample every prompt with the decoder still on CPU. The latents are a
            # few MB each, so parking them on CPU between phases costs nothing; the clip
            # forward gets the whole non-base headroom instead of sharing it with a decoder
            # it is not using yet.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()          # drop the training step's allocator slack
            # Clip forwards are big enough that VRAM pressure does not OOM on Windows - the
            # driver spills to system RAM and every step silently runs 3-6x slower (the
            # same failure signature as the checkpointing-margin bug). The fp32 Adam state
            # is ~2.5 GB of dead weight during a no-grad preview: park it on CPU for the
            # sampling phase. Costs about a second each way, once per preview epoch.
            if _frames > 1 and torch.cuda.is_available():
                for _st in optimizer.state.values():
                    for _k, _v in list(_st.items()):
                        if torch.is_tensor(_v) and _v.is_cuda:
                            _st[_k] = _v.to('cpu')
                            _opt_parked.append((_st, _k))
                if ema is not None:
                    # The fp32 shadow (~1.25 GB at LoKR factor 8) is dead weight during a
                    # no-grad preview — the EMA weights are already swapped INTO the live
                    # network. Park it with the optimizer state; next ema.update() is a
                    # training step away, after the finally restores it.
                    ema.shadow = [s.to('cpu') for s in ema.shadow]
                    _ema_parked = True
                if _opt_parked or _ema_parked:
                    gc.collect()
                    torch.cuda.empty_cache()
                _free0 = torch.cuda.mem_get_info()[0] / 1e9
                logger.info(f'[preview] clip sampling with {_free0:.1f} GB free '
                            f'({len(_opt_parked)} optimizer tensors parked'
                            f'{", EMA shadow parked" if _ema_parked else ""})')
            _rendered = []
            for i, txt in enumerate(_prompts):
                print(f"[preview] epoch {epoch}: prompt {i + 1}/{len(_prompts)} "
                      f"({_w}x{_h}, {_frames} frame(s), seed {_seed + i})", flush=True)
                lat = sampling.sample_image(
                    dit, txt.to(device, dtype),
                    width=_w, height=_h, steps=sample_steps,
                    cfg_scale=sample_cfg_scale,
                    uncond_embeds=(encoded_negative.to(device, dtype)
                                   if encoded_negative is not None else None),
                    seed=_seed + i, device=device, dtype=dtype, log_steps=True,
                    num_frames=_frames, on_slow_step=_slow_step_notice)
                _rendered.append((f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{_seed + i}",
                                  lat.to("cpu")))
                del lat

            # optimizer state back before anything else - the next training step needs it
            for _st, _k in _opt_parked:
                _st[_k] = _st[_k].to(device)
            _opt_parked = []

            # PHASE 2 — decode. Clip decode wants ~6 GB (decoder weights + chunk transients);
            # if the card cannot offer that next to the resident base, park the base on CPU
            # for the duration, exactly as the override-encode path does. A ~21 GB round trip
            # costs seconds once per preview epoch; an OOM used to cost the previews entirely.
            if decoder is not None:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    _free = torch.cuda.mem_get_info()[0] / 1e9
                    if _frames > 1 and _free < 7.5:
                        logger.info(f"[preview] {_free:.1f} GB free is too tight for clip "
                                    f"decode — parking the base on CPU for this decode pass.")
                        dit.to("cpu")
                        gc.collect()
                        torch.cuda.empty_cache()
                        _base_parked = True
                decoder = decoder.to(device)
            for stem, lat in _rendered:
                lat = lat.to(device)
                if lat.shape[2] > 1 and decoder is not None:
                    # Clip: decode every frame, store EVERY 2ND frame as JPEG in a sibling
                    # .clip dir, and save the MIDDLE frame as the contract PNG — written LAST,
                    # so the gallery/likeness settle guard sees one finished unit. The PNG name
                    # is the gallery/likeness/Visualiser contract; the .clip dir is additive.
                    px = decoder.decode_clip(lat.float())[0]     # [3, F, H, W] in [0, 1]
                    n_f = px.shape[1]
                    clip_dir = os.path.join(sample_dir, stem + ".clip")
                    os.makedirs(clip_dir, exist_ok=True)
                    _keep = list(range(0, n_f, 2))
                    if _keep[-1] != n_f - 1:
                        _keep.append(n_f - 1)          # always include the final frame
                    for k in _keep:
                        fr = (px[:, k].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                        Image.fromarray(fr).save(os.path.join(clip_dir, f"f{k:03d}.jpg"),
                                                 quality=87)
                    mid = (px[:, n_f // 2].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    img = Image.fromarray(mid)
                    print(f"[preview] decoded {n_f}-frame clip at {_w}x{_h} "
                          f"({(n_f + 1) // 2} scrub frames)", flush=True)
                    del px
                elif decoder is not None:
                    px = decoder.decode(lat.float())[0]          # [3, H, W] in [0, 1]
                    arr = (px.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
                    img = Image.fromarray(arr)
                    print(f"[preview] decoded {_w}x{_h}", flush=True)
                else:
                    # No VAE path configured — fall back to the 24ch->RGB linear approximation
                    # (a 1/16-scale rough look) rather than dropping previews entirely.
                    arr = sampling.latent_to_rgb(lat)
                    img = Image.fromarray(arr).resize((_w, _h), Image.NEAREST)
                    print(f"[preview] decoded {_w}x{_h}", flush=True)
                img.save(os.path.join(sample_dir, stem + ".png"))
                del lat
            if _base_parked:
                dit.to(device)
                if n_swap > 0:
                    dit.enable_block_swap(n_swap)     # restore the parked-block split
                gc.collect()
                torch.cuda.empty_cache()
                _base_parked = False
            logger.info(f"[preview] epoch {epoch}: wrote {len(_prompts)} sample(s) "
                        f"({sample_steps} steps, seed {_seed}) to {sample_dir}")
        finally:
            del decoder                                  # free the ~4.85 GB decoder immediately
            for _st, _k in _opt_parked:      # exception during phase 1: state must return
                _st[_k] = _st[_k].to(device)
            if _ema_parked:                  # next ema.update() needs the shadow on-device
                ema.shadow = [s.to(device) for s in ema.shadow]
            if _base_parked:
                # An exception mid-decode left the 21 GB base on CPU — the next training step
                # would die with "mat2 is on cpu". Restore residency (and the swap split)
                # before anything else runs.
                dit.to(device)
                if n_swap > 0:
                    dit.enable_block_swap(n_swap)
            if was_training:
                dit.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- epoch loop ----
    loss_recorder = LossRecorder()
    if do_previews and sample_at_first and start_epoch == 0:
        try:
            _render_previews(0)
        except Exception as _e0:
            if _clip_state["frames"] > 1:
                _was = _clip_state["frames"]
                _clip_state["frames"] = clip_fallback_frames(_was)
                logger.warning(
                    f"[preview] Sample at Start failed in CLIP mode ({type(_e0).__name__}) at "
                    f"{_was} frames — later previews retry at "
                    f"{_clip_state['frames']} frame(s). Training continues.")
            else:
                logger.warning(f"[preview] Sample at Start failed ({type(_e0).__name__}) — training "
                               f"continues; per-epoch previews will still be attempted.")
    progress_bar = tqdm(total=steps_per_epoch * max_train_epochs, initial=global_step,
                        desc="minimax-h3")
    # Distillation only: the two loss terms, summed over the epoch. The 0.8/0.2 weights are known
    # up front; what is not is how BIG each error is, and that is what actually decides how much
    # of the learning comes from real pixels versus from the teacher's rendering of them.
    _distill_parts = {}
    _distill_acc = [0.0, 0.0, 0]        # teacher sum, photo sum, count
    for epoch in range(start_epoch, max_train_epochs):
        shared_epoch.value = epoch + 1
        network.train()
        _distill_acc[:] = [0.0, 0.0, 0]
        # Identity-first: teacher-only while inside phase 1, photos-only after. Phase 2 takes the
        # ORDINARY loss path, so it never runs the teacher forward at all — half the compute of a
        # blended step, and no reference cache is touched.
        _teacher_phase = bool(distill and _p1_epochs and epoch < _p1_epochs)
        # Phase 1 runs at a THIRD of the Learning Rate box. It is placing the identity, not
        # reproducing detail, and it does that on a near-zero adapter where a full-size stride is
        # at its most destructive. Phase 2 gets the rate you actually asked for, starting from an
        # adapter that is already in the right place.
        _phase_lr = _P1_LR_SCALE if _teacher_phase else 1.0
        if _p1_epochs and epoch == _p1_epochs and epoch > start_epoch:
            logger.info(f"[distill] identity-first phase 1 complete after {_p1_epochs} epoch(s) "
                        f"— dropping the teacher; from here it trains on the photographs alone, "
                        f"at the full {learning_rate:.2e} (phase 1 ran at "
                        f"{learning_rate * _P1_LR_SCALE:.2e}).")
        for i, batch in enumerate(loader):
            latents = batch["latents"].to(device, dtype)           # (1, 24, H, W)
            if latents.dim() == 4:
                latents = latents.unsqueeze(2)                     # -> (1, 24, 1, H, W)
            text = batch["hidden_states"].to(device, dtype)        # (1, L, 5120)
            if uncond_text is not None and random.random() < caption_dropout:
                text = uncond_text.to(device, dtype)               # caption dropout step
            if distill and (_teacher_phase or not _p1_epochs) and "ref_hidden_states" in batch:
                _rz = batch["ref_latent"].to(device, dtype)      # (1, 24, h, w) from the cache
                if _rz.dim() == 4:
                    _rz = _rz.unsqueeze(2)                       # -> (1, 24, 1, h, w)
                loss, _ = compute_distill_loss(
                    dit, network, latents, text,
                    text_ref=batch["ref_hidden_states"].to(device, dtype),
                    ref_latents=[_rz],
                    text_token_tags=batch["ref_token_tags"][0],
                    # Phase 1 is teacher-ONLY (weight 1.0); the blended mode keeps the box value.
                    distill_weight=(1.0 if _teacher_phase else distill_weight),
                    shift=shift, seed=seed, parts_out=_distill_parts)
                _distill_acc[0] += _distill_parts["teacher"]
                _distill_acc[1] += _distill_parts["photo"]
                _distill_acc[2] += 1
            else:
                loss, _ = compute_loss(dit, latents, text, shift=shift)
            # Divide so the accumulated gradient is the MEAN over the window, not the sum —
            # otherwise the effective LR scales with the accumulation count.
            (loss / _accum_n if _accum_n > 1 else loss).backward()
            # Step on the window boundary, and always on the last batch of the epoch so a
            # partial tail window is never silently discarded.
            if (i + 1) % _accum_n == 0 or (i + 1) >= steps_per_epoch:
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                if warmup_steps or ramp is not None or _p1_epochs:
                    # Warmup, the ramp and the identity-first phase scale all COMPOSE: warmup
                    # covers the first steps (where the adapter is near zero and the ramp's ratio
                    # is undefined), the ramp takes over from there, and phase 1 runs the whole
                    # thing at a third of the box. The product is the LR.
                    _wf = (min(1.0, (global_step + 1) / warmup_steps) if warmup_steps else 1.0)
                    _rm = ramp.mult if ramp is not None else 1.0
                    for _g in optimizer.param_groups:
                        _g["lr"] = _g["_warmup_base_lr"] * _wf * _rm * _phase_lr
                if limiter is not None:
                    limiter.pre_step()   # snapshot BEFORE the optimizer moves anything
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if limiter is not None:
                    limiter.step()
                if ramp is not None:
                    ramp.step()          # post-clip weights; sets the multiplier for NEXT step
                if ema is not None:
                    ema.update()         # after the clip, so the shadow tracks clipped weights
            global_step += 1
            loss_recorder.add(epoch=epoch, step=i, loss=loss.item())
            progress_bar.set_postfix(avr_loss=f"{loss_recorder.moving_average:.4f}", refresh=False)
            progress_bar.update(1)

        logger.info(f"epoch {epoch + 1}/{max_train_epochs} done — avr_loss {loss_recorder.moving_average:.4f}")
        # Optimizer sanity: lora_up starts at zero and an Adam-family step is bounded by ~lr, so
        # after N steps no element can honestly exceed ~3*N*lr. When the 8-bit second moment
        # misbehaves (v quantized to zero -> update degrades to lr*m/eps) the drift blows through
        # that bound by orders of magnitude — caught here per epoch instead of per melted preview.
        try:
            _lr_now = optimizer.param_groups[0]["lr"]
            _drift = max((float(l.lora_up.weight.detach().abs().max())
                          for l in network.unet_loras if hasattr(l, "lora_up")), default=0.0)
            _bound = 3.0 * global_step * _lr_now
            if _drift > _bound:
                logger.warning(f"[drift] max|lora_up|={_drift:.4f} EXCEEDS the Adam bound "
                               f"~{_bound:.4f} ({global_step} steps @ lr={_lr_now:.1e}) — the "
                               f"optimizer is stepping far beyond the configured LR (8-bit "
                               f"state underflow?). Expect degraded samples.")
            else:
                logger.info(f"[drift] max|lora_up|={_drift:.4f} (bound ~{_bound:.4f} — healthy)")
        except Exception:
            pass
        if _p1_epochs and not _teacher_phase and distill:
            logger.info(f"[distill] photos only (identity-first phase 2) — the teacher was "
                        f"dropped after epoch {_p1_epochs}.")
        if _distill_acc[2]:
            _t = _distill_acc[0] / _distill_acc[2]
            _p = _distill_acc[1] / _distill_acc[2]
            _w = 1.0 if _teacher_phase else float(distill_weight)
            # Weighted contributions are what the optimizer actually sees. The raw errors are
            # printed too, because the interesting question is whether the photo term is HARDER
            # (bigger error) than the teacher term, which is what lets 20% punch above its weight.
            _wt, _wp = _w * _t, (1.0 - _w) * _p
            _tot = _wt + _wp
            logger.info(
                f"[distill] teacher err {_t:.4f} x{_w:.2f} = {_wt:.4f} | "
                f"photo err {_p:.4f} x{1 - _w:.2f} = {_wp:.4f} | "
                f"real pixels are {100 * _wp / _tot if _tot else 0:.0f}% of this epoch's loss "
                f"(the weight alone says {100 * (1 - _w):.0f}%)")
        if limiter is not None:
            logger.info(limiter.epoch_report())
        if ramp is not None:
            logger.info(ramp.epoch_report())
        if adaptive is not None:
            adaptive.epoch_boundary(epoch, loss_recorder.moving_average, network, optimizer)
        if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0 and (epoch + 1) < max_train_epochs:
            ckpt = os.path.join(output_dir, f"{output_name}-{epoch + 1:06d}.safetensors")
            if ema is not None:
                ema.swap_in()
            try:
                _save_lora(network, ckpt, network_dim, network_alpha, dtype, _meta())
            finally:
                if ema is not None:
                    ema.swap_out()
            logger.info(f"saved {ckpt}")
            if save_state:
                # Non-fatal (see the krea2 twin): a failed convenience save must never kill a
                # run whose checkpoint already wrote. Truly-full disks fail the next epoch
                # CHECKPOINT, and that one is rightly fatal.
                try:
                    _save_training_state(output_dir, output_name, network, optimizer,
                                         epoch=epoch + 1, global_step=global_step,
                                         dtype=dtype, extra=_state_extra(), ema=ema)
                    prune_state_dirs(output_dir, output_name, keep_last_n_states)
                except Exception as _se:
                    logger.error("[state] saving the resume state FAILED (%s: %s) — likely the "
                                 "disk (on RunPod the volume quota is only visible in the "
                                 "dashboard). Training continues; this epoch has no resume "
                                 "point. The epoch checkpoint itself already saved.",
                                 type(_se).__name__, _se)
        if do_previews and sample_every_n_epochs and (epoch + 1) % sample_every_n_epochs == 0:
            try:
                # Previews render on the EMA weights when EMA is on — a preview must show what
                # the saved checkpoint will look like, not the raw zigzag the EMA exists to hide.
                if ema is not None:
                    ema.swap_in()
                try:
                    _render_previews(epoch + 1)
                finally:
                    if ema is not None:
                        ema.swap_out()
            except Exception as _pe:
                # Latch previews OFF for the rest of the run rather than re-failing (and
                # re-OOMing) every epoch. Training and checkpoints are never at risk.
                _oom = "out of memory" in str(_pe).lower()
                if _clip_state["frames"] > 1:
                    # The failure arrived in CLIP mode — the mode a still preview never
                    # exercised. Step DOWN the frame grid and keep clips rather than ending
                    # every preview for the run; only a failure at stills latches off.
                    _was = _clip_state["frames"]
                    _clip_state["frames"] = clip_fallback_frames(_was)
                    logger.warning(
                        f"[preview] epoch {epoch + 1} CLIP preview failed at {_was} frames "
                        f"({'CUDA OOM' if _oom else type(_pe).__name__}) — retrying at "
                        f"{_clip_state['frames']} frame(s) from the next preview on. Training "
                        f"continues and LoRAs still save normally.")
                else:
                    logger.warning(
                        f"[preview] epoch {epoch + 1} preview failed "
                        f"({'CUDA OOM' if _oom else type(_pe).__name__}); disabling previews for "
                        f"the rest of the run. Training continues and LoRAs still save normally.")
                    do_previews = False
            network.train()
        if os.path.exists(pause_flag):
            # Pause = graceful epoch-end exit with FULL state (regardless of the save-state
            # toggles), so Resume continues exactly here — matching Klein/Krea 2. The final
            # LoRA is deliberately NOT written; Resume (or the natural run end) writes it.
            try:
                _save_training_state(output_dir, output_name, network, optimizer,
                                     epoch=epoch + 1, global_step=global_step,
                                     dtype=dtype, extra=_state_extra(), ema=ema)
            except Exception as _se:
                logger.error("[pause] state save FAILED (%s: %s) — there is NO new resume point "
                             "for this pause. Free disk space (RunPod: dashboard quota) and "
                             "resume from the previous saved state.", type(_se).__name__, _se)
            try:
                os.remove(pause_flag)
            except OSError:
                pass
            progress_bar.close()
            logger.info(f"[pause] requested — state saved at epoch {epoch + 1}. Exiting cleanly.")
            sys.exit(0)

    progress_bar.close()
    final = os.path.join(output_dir, f"{output_name}.safetensors")
    if ema is not None:
        ema.swap_in()
    try:
        _save_lora(network, final, network_dim, network_alpha, dtype, _meta())
    finally:
        if ema is not None:
            ema.swap_out()
    logger.info(f"saved final LoRA: {final}")
    if save_state_on_train_end and max_train_epochs > start_epoch:
        # Non-fatal: the final LoRA is already on disk; dying here would turn a finished run red.
        try:
            _save_training_state(output_dir, output_name, network, optimizer,
                                 epoch=max_train_epochs, global_step=global_step,
                                 dtype=dtype, extra=_state_extra(), ema=ema)
            prune_state_dirs(output_dir, output_name, keep_last_n_states)
        except Exception as _se:
            logger.error("[state] end-of-run state save FAILED (%s: %s) — the finished LoRA is "
                         "saved and fine; only train-further-by-resume is affected.",
                         type(_se).__name__, _se)
    try:
        os.remove(pause_flag)
    except OSError:
        pass
    return final
