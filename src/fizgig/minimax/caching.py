"""MiniMax H3 caching: video-VAE latents (24-ch) + Qwen3-VL-32B text-encoder states.

Reuses Fizgig's dataset framework (ItemInfo + safetensors cache files, same as Krea 2); only
the encode + the cache contents are H3-specific:

* Latents: the H3 video VAE encodes a still image as one frame -> [1, 24, 1, H/16, W/16]. The
  1-frame axis is squeezed for storage -> (24, H/16, W/16), keyed `latent_{H}x{W}` so the shared
  collate maps it to the batch's `latents`. The trainer re-adds the frame axis before the DiT.
* Text: the H3 conditioning is the raw layer-50 Qwen3-VL output -> (L, 5120). Stored as
  `hidden_states` (L, 5120) plus an all-valid `attention_mask` (L,), matching the Krea 2 text
  cache keys so the same collate path applies (batch size 1 -> no padding needed).
"""

import logging
import os
from typing import List

import torch
from safetensors.torch import save_file

from fizgig.dataset.image_dataset import ItemInfo, MemoryEfficientSafeOpen, dtype_to_str
from fizgig.training.metadata import ARCHITECTURE_MINIMAX

logger = logging.getLogger(__name__)


# --- cache writers -----------------------------------------------------------
def save_latent_cache_minimax(item_info: ItemInfo, latent: torch.Tensor) -> None:
    """Save an H3 VAE latent (3-D: C, H, W) to the item's latent cache file."""
    assert latent.dim() == 3, f"latent must be 3-D (C,H,W), got {tuple(latent.shape)}"
    _, H, W = latent.shape
    sd = {f"latent_{H}x{W}": latent.detach().cpu().contiguous()}
    for key, value in sd.items():
        if torch.isnan(value).any():
            logger.warning(f"NaN in {key} for {item_info.item_key} - replaced with 0")
            value[torch.isnan(value)] = 0
    metadata = {
        "architecture": ARCHITECTURE_MINIMAX,
        "width": str(item_info.original_size[0]),
        "height": str(item_info.original_size[1]),
        "dtype": dtype_to_str(latent.dtype),
        "format_version": "2.0.0",
    }
    os.makedirs(os.path.dirname(item_info.latent_cache_path), exist_ok=True)
    save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache_minimax(item_info: ItemInfo, hidden_states: torch.Tensor) -> None:
    """Save the raw Qwen3-VL-32B layer-50 states (L, 5120) + an all-valid mask (L,)."""
    hidden_states = hidden_states.detach().cpu().contiguous()
    if torch.isnan(hidden_states).any():
        logger.warning(f"NaN in hidden_states for {item_info.item_key} - replaced with 0")
        hidden_states[torch.isnan(hidden_states)] = 0
    sd = {
        "hidden_states": hidden_states,
        "attention_mask": torch.ones(hidden_states.shape[0], dtype=torch.bool),
    }
    metadata = {
        "architecture": ARCHITECTURE_MINIMAX,
        "caption1": item_info.caption,
        "dtype": dtype_to_str(hidden_states.dtype),
        "format_version": "2.0.0",
    }
    os.makedirs(os.path.dirname(item_info.text_encoder_output_cache_path), exist_ok=True)
    save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata)


def reference_cache_path(te_cache_path: str, slot: int = 0) -> str:
    """One r2v conditioning slot beside an item's ordinary text cache.

    `..._minimaxh3_te.safetensors` -> `..._minimaxh3_teref0.safetensors` (slot 0, 1, ...).

    SIBLING files, never a replacement: a distillation run needs both, since the student is
    conditioned on the plain caption and only the teacher gets the reference. Several slots
    because each item is paired with several OTHER dataset images — one reference per slot,
    picked at random per step, so the teacher's sequence stays short while the LoRA still sees
    every pairing across an epoch.

    The dataset's stale-cache glob is `*_{arch}_te.safetensors`, which matches none of these."""
    stem, ext = os.path.splitext(te_cache_path)
    return f"{stem}ref{int(slot)}{ext}"


def reference_cache_slots(te_cache_path: str, max_slots: int = 8):
    """Which reference slots actually exist for an item, as a list of ints."""
    return [i for i in range(max_slots) if os.path.exists(reference_cache_path(te_cache_path, i))]


def save_reference_text_cache(te_cache_path: str, slot: int, hidden_states: torch.Tensor,
                              token_tags: torch.Tensor, ref_latent: torch.Tensor,
                              caption: str = "", reference: str = "") -> str:
    """Save one slot: r2v conditioning (L, 5120), its per-row tags (L,), and the reference's
    own latent (24, h, w).

    All three travel together because they only mean anything together. The tags say which rows
    are the `<Picture 1>` vision block — states without them would have those rows silently
    modulated as text. The latent must be the SAME picture the vision block describes, or the
    teacher is shown one image and conditioned on another."""
    hidden_states = hidden_states.detach().cpu().contiguous()
    if torch.isnan(hidden_states).any():
        logger.warning(f"NaN in reference hidden_states for {te_cache_path} - replaced with 0")
        hidden_states[torch.isnan(hidden_states)] = 0
    path = reference_cache_path(te_cache_path, slot)
    sd = {"hidden_states": hidden_states,
          "token_tags": token_tags.detach().cpu().to(torch.int64).contiguous(),
          "ref_latent": ref_latent.detach().cpu().contiguous()}
    metadata = {"architecture": ARCHITECTURE_MINIMAX, "caption1": caption,
                "reference": os.path.basename(reference), "slot": str(slot),
                "dtype": dtype_to_str(hidden_states.dtype), "format_version": "2.0.0"}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(sd, path, metadata=metadata)
    return path


def load_reference_text_cache(te_cache_path: str, slot: int = 0):
    """-> (hidden_states, token_tags, ref_latent) for a slot, or None when it does not exist."""
    path = reference_cache_path(te_cache_path, slot)
    if not os.path.exists(path):
        return None
    # MemoryEfficientSafeOpen, not safetensors.safe_open — the repo uses it everywhere for the
    # documented Windows mmap crash, and these are read on the training hot path.
    with MemoryEfficientSafeOpen(path) as f:
        return (f.get_tensor("hidden_states"), f.get_tensor("token_tags"),
                f.get_tensor("ref_latent"))


def load_text_encoder_output_cache_minimax(path: str):
    """Return (hidden_states (L, 5120), attention_mask (L,) bool) from a cache file."""
    with MemoryEfficientSafeOpen(path) as f:
        return f.get_tensor("hidden_states"), f.get_tensor("attention_mask").to(torch.bool)


# --- batch encoders (called by the cache scripts) ----------------------------
def encode_and_save_latents(vae, batch: List[ItemInfo]) -> None:
    """Encode an image batch through the H3 video VAE and save each 24-ch latent."""
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    contents = []
    for item in batch:
        content = item.content
        content = content[0] if isinstance(content, list) else content  # (H, W, C) uint8
        contents.append(torch.from_numpy(content))
    contents = torch.stack(contents, dim=0).permute(0, 3, 1, 2)  # (B, C, H, W)
    contents = contents / 127.5 - 1.0                            # -> [-1, 1]
    with torch.no_grad():
        latents = vae.encode(contents.to(device, dtype=dtype))  # (B, 24, 1, H/16, W/16)
    for b, item in enumerate(batch):
        latent = latents[b]                        # (24, 1, H/16, W/16)
        latent = latent.squeeze(1) if latent.dim() == 4 else latent  # -> (24, H/16, W/16)
        logger.info(f"latent cache: {item.item_key} -> {tuple(latent.shape)}")
        save_latent_cache_minimax(item, latent)


@torch.no_grad()
def encode_and_save_reference_text(encoder, batch: List[ItemInfo], reference_for) -> None:
    """Encode each item's caption against its assigned reference images, one file per slot.

    `reference_for(item)` returns [(slot, PIL image, latent, name), ...] — the pairing lives in
    the caching script, which is the only place that can see the whole dataset.

    One reference per encode rather than several packed together: the total cache is the same,
    but the teacher's sequence at training time stays short."""
    for item in batch:
        for slot, image, latent, name in reference_for(item):
            hidden, tags = encoder.encode_with_reference(item.caption, [image])
            save_reference_text_cache(item.text_encoder_output_cache_path, slot, hidden[0], tags,
                                      latent, caption=item.caption, reference=name)


def encode_and_save_text(encoder, batch: List[ItemInfo], batch_size: int = None) -> None:
    """Encode a caption batch through the Qwen3-VL-32B TE and save each (L, 5120) state.

    One forward for the whole batch rather than one per caption: the nvfp4-resident encoder
    dequantizes 351 weights per FORWARD, so this is where the caching pass spends its time.
    encode_batch right-pads, which is exactly equivalent for a causal stack (verified in
    tests/diag_batch_encode.py) and memoizes, so a repeated caption costs nothing.

    batch_size is the captions per FORWARD. Passing it through matters: encode_batch re-chunks
    internally at its own default of 8, so without this the caller's --batch_size only decided
    how many items arrived here, never how many went through the model at once."""
    kw = {} if batch_size is None else {"batch_size": int(batch_size)}
    embeds = encoder.encode_batch([item.caption for item in batch], **kw)
    for item, hidden in zip(batch, embeds):
        save_text_encoder_output_cache_minimax(item, hidden[0])
