"""RepairEngine — owns the Klein pipeline and primary/donor LoRA networks.

Single preview entry point: generate_preview(state) -> PIL.Image. v2 will
replace the body to consult an activation cache; the UI never reaches around
this method.
"""

import json
import logging
import os
import re
from typing import List, Optional, Set

from PIL import Image

from fizgig.repair_studio.state import SliderState, block_regex

logger = logging.getLogger(__name__)


def _slerp(t, a, b):
    """Spherical linear interpolation between two noise tensors at fraction t
    (0 -> a, 1 -> b). Walks the hypersphere so the interpolated noise keeps a
    constant norm — lerp would shrink it mid-way and the sample would go mushy.
    Falls back to lerp for (anti)parallel endpoints."""
    import torch
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    na, nb = a_f.norm(), b_f.norm()
    if na.item() == 0.0 or nb.item() == 0.0:
        out = (1.0 - t) * a_f + t * b_f
    else:
        dot = torch.dot(a_f / na, b_f / nb).clamp(-1.0, 1.0)
        omega = torch.acos(dot)
        so = torch.sin(omega)
        if so.item() < 1e-6:
            out = (1.0 - t) * a_f + t * b_f
        else:
            out = (torch.sin((1.0 - t) * omega) / so) * a_f + (torch.sin(t * omega) / so) * b_f
    return out.reshape(a.shape).to(a.dtype)


def find_profile_for_hash(profiles_dir: str, target_hash: str) -> Optional[dict]:
    """Scan `profiles_dir` for *.json sidecars written by the Profiler and
    return the first payload whose `hash` field matches `target_hash`.

    Returns None if no directory, no match, or any parse error. Designed to
    be cheap enough to call from a GUI event handler — the directory is
    small (one JSON per profiled LoRA) and each JSON is tiny.
    """
    if not target_hash or not profiles_dir or not os.path.isdir(profiles_dir):
        return None
    try:
        candidates = sorted(fn for fn in os.listdir(profiles_dir) if fn.endswith(".json"))
    except Exception:
        return None
    for fn in candidates:
        path = os.path.join(profiles_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("hash") == target_hash:
            payload["_sidecar_path"] = path
            return payload
    return None


_LORA_BLOCK_RX = re.compile(r"(double|single)_blocks_(\d+)_")


def _extract_block_ids_from_network(network) -> Set[str]:
    """Return the set of block_ids (e.g. {'single_5', 'double_3'}) that have
    LoRA modules in this network. An Identity-only LoRA typically returns a
    subset like {'single_1'..'single_16'}; a full-model LoRA returns all 32.
    """
    ids: Set[str] = set()
    if network is None:
        return ids
    for mod in network.unet_loras:
        m = _LORA_BLOCK_RX.search(mod.lora_name)
        if m:
            ids.add(f"{m.group(1)}_{int(m.group(2))}")
    return ids


class RepairEngine:
    def __init__(self):
        self.pipeline = None
        self.primary_network = None
        self.donor_network = None
        self.primary_path: Optional[str] = None
        self.donor_path: Optional[str] = None

        # Set of block_ids present in each network. Lets the UI grey out
        # sliders for blocks the LoRA doesn't actually touch.
        self.primary_block_ids: Set[str] = set()
        self.donor_block_ids: Set[str] = set()

        # SHA-256 of the primary LoRA's bytes — used by the GUI to cross-link
        # to any matching Profiler report sidecar in profiles_dir.
        self.primary_hash: Optional[str] = None

        # v2 hook — set by UI before generate_preview.
        self._changed_blocks: set = set()

        # v2 Turbo Preview — activation cache
        self._turbo_enabled: bool = False
        self._act_cache: Optional[dict] = None       # timestep_idx → ActivationCacheEntry
        self._act_cache_key: Optional[tuple] = None   # (primary_path, donor_path, seed, prompt, w, h)
        self._act_cache_state: Optional[SliderState] = None  # the state the cache reflects (resume diff)

        # Prompt encoding cache — avoids reloading TE + re-encoding on slider-only changes
        self._prompt_cache_key: Optional[tuple] = None   # (prompt, primary_path)
        self._prompt_cache: Optional[tuple] = None       # (ctx_vec, neg_ctx_vec)

        # Reference-image encode cache — avoids re-running the VAE on slider tweaks
        self._ref_cache_key: Optional[tuple] = None   # (path, megapixels, strength)
        self._ref_cache: Optional[tuple] = None       # (ref_tokens, ref_ids)

        # Clean latent of the most recent generate_preview ([128, H/16, W/16]).
        # Sequential travel feeds this back as the next frame's reference latent.
        self._last_frame_latent = None

        # Cache the last-generated baseline keyed on (primary_path, seed,
        # prompt, w, h) — only regenerate baseline when these change.
        self._baseline_cache_key = None
        self._baseline_cache_image: Optional[Image.Image] = None

    # ------------------------------------------------------------------
    # Pipeline + LoRA loading
    # ------------------------------------------------------------------

    def ensure_pipeline(
        self,
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        model_version: str = "klein-9b",
        device: str = "cuda",
        fp8_scaled: bool = False,
        fp8_text_encoder: bool = True,
        blocks_to_swap: int = 0,
        int8: bool = False,
    ) -> None:
        """Lazy-load the Klein pipeline. No-op if already loaded.

        fp8_text_encoder defaults to True — Qwen3-8B in bf16 is ~16GB; fp8 is ~8GB,
        keeping room for the bf16 DiT (~18GB) on a 32GB card.
        """
        if self.pipeline is not None and self.pipeline.is_loaded:
            return

        from fizgig.klein.inference import KleinInferencePipeline

        self.pipeline = KleinInferencePipeline()
        self.pipeline.load_models(
            dit_path=dit_path,
            vae_path=vae_path,
            text_encoder_path=text_encoder_path,
            model_version=model_version,
            device=device,
            fp8_scaled=fp8_scaled,
            fp8_text_encoder=fp8_text_encoder,
            blocks_to_swap=blocks_to_swap,
            int8=int8,
        )
        # Offload TE to CPU between calls (saves ~8GB). generate_preview
        # reloads it to GPU before each encode.
        try:
            self.pipeline.unload_text_encoder()
        except Exception:
            logger.exception("Failed to offload text encoder (non-fatal)")
        logger.info("Repair Studio pipeline ready (model_version=%s, fp8_te=%s)",
                    model_version, fp8_text_encoder)

    def load_primary(self, path: str) -> None:
        """Load primary LoRA without merging — keeps multipliers live.

        v1 limitation: changing primary requires reset() first. The DiT's
        forward methods are patched on apply_to(); we don't unwind those.
        """
        if self.pipeline is None or not self.pipeline.is_loaded:
            raise RuntimeError("Pipeline not loaded; call ensure_pipeline() first.")
        if self.primary_network is not None:
            raise RuntimeError("Primary already loaded — call reset() to swap.")

        self.pipeline.load_lora_for_profiling(path, multiplier=1.0)
        self.primary_network = self.pipeline._lora_network
        self.primary_path = path
        self.primary_block_ids = _extract_block_ids_from_network(self.primary_network)
        # Content hash for Profiler cross-link (GUI looks up the sidecar).
        from fizgig.profiler.visualize import compute_lora_hash
        self.primary_hash = compute_lora_hash(path)
        self._invalidate_baseline_cache()
        self._invalidate_activation_cache()
        logger.info("Repair Studio primary loaded: %s (blocks: %d, hash: %s)",
                    path, len(self.primary_block_ids),
                    self.primary_hash[:12] + "…" if self.primary_hash else "?")

    def swap_primary_weights(self, path: str) -> bool:
        """LoRA Royale fast path: swap the primary network's weights to another
        checkpoint of the SAME structure (epochs of one run) WITHOUT reloading
        the pipeline or re-patching the DiT — load_state_dict in place.

        Returns True on a clean swap, False if the checkpoint's structure doesn't
        match the patched network (e.g. an unrelated LoRA / different rank), in
        which case the caller should reset() + load_primary() instead.
        """
        import torch
        if self.primary_network is None:
            raise RuntimeError("No primary loaded; call load_primary() first.")
        from fizgig.networks.lora import ensure_kohya_lora_state_dict
        from safetensors.torch import load_file
        try:
            sd = ensure_kohya_lora_state_dict(load_file(path))
        except Exception:
            logger.exception("swap_primary_weights: failed to load %s", path)
            return False
        try:
            info = self.primary_network.load_state_dict(sd, strict=False)
        except Exception as e:
            # strict=False tolerates missing/extra keys but NOT shape mismatches —
            # a different-rank checkpoint raises "size mismatch ...". Bail so the
            # caller does a full reset()+load_primary() that rebuilds for the new rank.
            logger.info("swap_primary_weights: structure/shape mismatch, needs full reload (%s)", e)
            return False
        # If most of the checkpoint's keys didn't map onto the patched network,
        # the structure differs — bail so the caller can do a full reload.
        if sd and len(info.unexpected_keys) > 0.5 * len(sd):
            return False
        self.primary_network.to(self.pipeline.device, dtype=torch.bfloat16)
        self.primary_path = path
        try:
            from fizgig.profiler.visualize import compute_lora_hash
            self.primary_hash = compute_lora_hash(path)
        except Exception:
            self.primary_hash = None
        self._invalidate_baseline_cache()
        self._invalidate_activation_cache()
        return True

    def load_donor(self, path: str) -> None:
        """Load a second LoRA on top (chained, additive). All blocks start disabled."""
        if self.pipeline is None or not self.pipeline.is_loaded:
            raise RuntimeError("Pipeline not loaded.")
        if self.primary_network is None:
            raise RuntimeError("Load primary LoRA before donor.")
        if self.donor_network is not None:
            raise RuntimeError("Donor already loaded — call reset() or unload_donor() first.")

        from fizgig.networks.lora_klein import create_arch_network_from_weights
        from fizgig.networks.lora import ensure_kohya_lora_state_dict
        from safetensors.torch import load_file

        weights_sd = load_file(path)
        weights_sd = ensure_kohya_lora_state_dict(weights_sd)

        network = create_arch_network_from_weights(
            multiplier=1.0,
            weights_sd=weights_sd,
            unet=self.pipeline.dit,
            for_inference=True,
        )
        # apply_to chains forwards on top of the primary patches.
        network.apply_to(
            text_encoders=None,
            unet=self.pipeline.dit,
            apply_text_encoder=False,
            apply_unet=True,
        )
        network.load_state_dict(weights_sd, strict=False)
        network.to(self.pipeline.device, dtype=self.pipeline.dit_dtype)

        # Disable all donor modules — they only contribute when the user
        # explicitly enables them per-block via the slider state.
        network.set_enabled(False)

        self.donor_network = network
        self.donor_path = path
        self.donor_block_ids = _extract_block_ids_from_network(network)
        self._invalidate_activation_cache()
        logger.info("Repair Studio donor loaded: %s (%d modules, %d blocks)",
                    path, len(network.unet_loras), len(self.donor_block_ids))

    def unload_donor(self) -> None:
        """Disable donor (modules stay patched but transparent). Cheap; no DiT reload."""
        if self.donor_network is not None:
            self.donor_network.set_enabled(False)
            self.donor_network = None
            self.donor_path = None
            self.donor_block_ids = set()
            self._invalidate_activation_cache()
            logger.info("Donor unloaded (chain remains; reset() for full cleanup)")

    def reset(self) -> None:
        """Full unload — drops pipeline, primary, donor. Next ensure_pipeline reloads from disk."""
        import gc
        import torch
        from fizgig.utils.device import clean_memory_on_device

        import gc
        import torch

        # Invalidate caches first (drops large GPU tensors from turbo cache).
        self._invalidate_baseline_cache()
        self._invalidate_activation_cache()

        # Drop LoRA networks BEFORE the pipeline — their forward-hook closures
        # capture `org_forward` (a bound method on the DiT's Linear layers),
        # creating circular references that prevent GC from freeing the DiT.
        # Explicitly clear the module lists to break the cycles.
        from fizgig.utils.device import release_module_tensors as _strip
        for net in (self.primary_network, self.donor_network):
            if net is not None:
                try:
                    _strip(net)     # husk the LoRA weights — see the H3 reset comment
                    for lora in net.unet_loras:
                        lora.org_forward = None
                    net.unet_loras.clear()
                    for lora in net.text_encoder_loras:
                        lora.org_forward = None
                    net.text_encoder_loras.clear()
                except Exception:
                    pass
        self.primary_network = None
        self.donor_network = None

        # Force GC to break any remaining ref cycles before pipeline unload.
        gc.collect()

        if self.pipeline is not None:
            # Same defense as the H3/Krea 2 engines: husk the DiT before dropping it, so an
            # externally-pinned module graph can't keep the VRAM (field leak, 19 Aug).
            from fizgig.utils.device import release_module_tensors
            release_module_tensors(getattr(self.pipeline, "dit", None))
            try:
                self.pipeline.unload_models()
            except Exception:
                logger.exception("Pipeline unload raised; continuing")
        self.pipeline = None

        self.primary_path = None
        self.donor_path = None
        self.primary_block_ids = set()
        self.donor_block_ids = set()
        self.primary_hash = None
        self._changed_blocks.clear()
        self._prompt_cache_key = None
        self._prompt_cache = None
        self._ref_cache_key = None
        self._ref_cache = None

        # Final GC + CUDA cache flush.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from fizgig.utils.device import report_cuda_leak, flush_reserved_vram
        report_cuda_leak("klein-repair-reset")
        flush_reserved_vram("klein-repair-reset")

    # ------------------------------------------------------------------
    # Activation cache (v2 Turbo Preview)
    # ------------------------------------------------------------------

    def _invalidate_activation_cache(self):
        """Release all cached activation tensors."""
        self._act_cache = None
        self._act_cache_key = None
        self._act_cache_state = None

    def _compute_resume_point(self, changed) -> Optional[tuple]:
        """Earliest block to resume from, given the list of changed block ids (a state diff).
        Doubles run before singles, so any changed double resumes from that double; otherwise
        the earliest changed single."""
        if not changed:
            return None
        doubles = sorted(int(b.split("_")[1]) for b in changed if b.startswith("double_"))
        singles = sorted(int(b.split("_")[1]) for b in changed if b.startswith("single_"))
        if doubles:
            return ("double", doubles[0])
        if singles:
            return ("single", singles[0])
        return None

    # ------------------------------------------------------------------
    # State application
    # ------------------------------------------------------------------

    def apply_state(self, state: SliderState) -> None:
        """Push the slider state into the live LoRA networks."""
        if self.primary_network is None:
            raise RuntimeError("No primary LoRA loaded.")

        for block_id, bs in state.blocks.items():
            pat = block_regex(block_id)
            self.primary_network.set_module_enabled_by_pattern(pat, bs.primary_enabled, target="unet")
            self.primary_network.set_module_multiplier_by_pattern(pat, bs.primary_strength, target="unet")
            if self.donor_network is not None:
                self.donor_network.set_module_enabled_by_pattern(pat, bs.donor_enabled, target="unet")
                self.donor_network.set_module_multiplier_by_pattern(pat, bs.donor_strength, target="unet")

    def mark_blocks_changed(self, block_ids: List[str]) -> None:
        # No-op compatibility shim: the resume point is now derived from a state diff against
        # _act_cache_state at render time (race-free), not a mutable changed-set that could be
        # cleared mid-render and drop edits made while a preview was in flight.
        pass

    # ------------------------------------------------------------------
    # Generation — the single preview entry point (v2 replaces body only)
    # ------------------------------------------------------------------

    def _build_ref_tokens(self, state: SliderState, prev_latent=None,
                          prev_latent_strength=1.0):
        """Encode reference image(s) into (ref_tokens, ref_ids) for edit-conditioning.

        Klein is an edit model and supports MULTIPLE references: the primary
        `ref_image_path`, plus an optional second `ref2_path`. Each is VAE-encoded,
        scaled by its strength, and packed together — pack_control_latent gives
        each its own position offset (the "index"/Independent method). The dual-ref
        case is used by sequential travel: ref = original (clean identity anchor),
        ref2 = previous frame (temporal continuity) — the original re-injects clean
        detail every frame so VAE feedback drift can't accumulate.

        `prev_latent` (optional): the previous frame's CLEAN latent ([128, H/16,
        W/16], as cached by generate_preview). When supplied it is packed directly
        as the previous-frame reference — skipping the lossy decode→PNG→encode round
        trip the image path incurs. This is what keeps sequential travel sharp at
        high strength. It is appended LAST (image 1 = original anchor, image 2 =
        previous-frame latent), and the token cache is bypassed (it changes per frame).

        Caps each image to ref_megapixels (downscale only, ×16). The ref latent goes
        in the POSITIVE conditioning (caller concatenates onto the cond pass only).
        Returns (None, None) when no ref is set."""
        mp = max(0.05, float(getattr(state, "ref_megapixels", 1.0) or 1.0))
        # Collect image references in order: primary, then optional second anchor.
        refs = []
        p1 = (getattr(state, "ref_image_path", "") or "").strip()
        s1 = float(getattr(state, "ref_strength", 1.0))
        if p1 and os.path.exists(p1) and s1 != 0.0:
            refs.append((p1, s1))
        p2 = (getattr(state, "ref2_path", "") or "").strip()
        s2 = float(getattr(state, "ref2_strength", 1.0))
        if p2 and os.path.exists(p2) and s2 != 0.0:
            refs.append((p2, s2))
        use_prev_latent = (prev_latent is not None
                           and float(prev_latent_strength) != 0.0)
        if not refs and not use_prev_latent:
            return None, None

        # Cache the packed tokens for the exact image ref set (skip re-encode on
        # slider tweaks). A latent ref changes every frame, so bypass the cache then.
        ref_key = (tuple((p, round(s, 4)) for p, s in refs), round(mp, 4))
        if (not use_prev_latent and self._ref_cache_key == ref_key
                and self._ref_cache is not None):
            return self._ref_cache

        import numpy as np
        import torch
        from PIL import Image as _PILImage
        from fizgig.klein.position import pack_control_latent
        pipeline = self.pipeline

        # Encode on the pipeline device (VAE may be CPU-offloaded between previews);
        # move it on for the encodes, then restore so we don't disturb VRAM planning.
        vae_dev = next(pipeline.vae.parameters()).device
        pipeline.vae.to(pipeline.device).eval()
        latents = []
        try:
            for path, strength in refs:
                img = _PILImage.open(path).convert("RGB")
                w, h = img.size
                cur_mp = (w * h) / 1_000_000.0
                if cur_mp > mp:  # downscale only (matches ReferenceLatentPlus _cap_megapixels)
                    sc = (mp / cur_mp) ** 0.5
                    w, h = int(w * sc), int(h * sc)
                w = max(16, (w // 16) * 16)
                h = max(16, (h // 16) * 16)
                img = img.resize((w, h), _PILImage.BICUBIC)
                arr = np.asarray(img, dtype=np.float32)  # H, W, 3 in [0, 255]
                t = torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0  # C, H, W in [-1, 1]
                t = t.unsqueeze(0).to(pipeline.device, dtype=pipeline.vae.dtype)
                with torch.no_grad():
                    lat = pipeline.vae.encode(t)[0]  # [128, H/16, W/16] (packed)
                if strength != 1.0:
                    lat = lat * strength
                latents.append(lat.to(pipeline.device))
        finally:
            if vae_dev.type != pipeline.device.type:
                pipeline.vae.to(vae_dev)

        # Previous-frame latent goes in directly (no VAE round trip), last in order
        # so it reads as "image 2" after any clean original anchor.
        if use_prev_latent:
            lat = prev_latent.to(pipeline.device, dtype=pipeline.vae.dtype)
            if float(prev_latent_strength) != 1.0:
                lat = lat * float(prev_latent_strength)
            latents.append(lat)

        ref_tokens, ref_ids = pack_control_latent(latents)
        ref_tokens = ref_tokens.to(device=pipeline.device, dtype=torch.bfloat16)
        ref_ids = ref_ids.to(pipeline.device)

        if not use_prev_latent:
            self._ref_cache_key = ref_key
            self._ref_cache = (ref_tokens, ref_ids)
        return ref_tokens, ref_ids

    def encode_travel_prompts(self, prompts):
        """Encode a list of waypoint prompts for prompt travel. Reloads the
        text encoder once, encodes every prompt, then unloads it. Returns
        (list_of_ctx_vec, neg_ctx_vec) with the conditioning tensors on CPU
        (cheap to hold many; moved back to device per render). The negative
        conditioning is encoded once and shared."""
        import torch
        from fizgig.utils.device import clean_memory_on_device
        if self.pipeline is None or not self.pipeline.is_loaded:
            raise RuntimeError("Pipeline not loaded.")
        pipeline = self.pipeline
        clean_memory_on_device(pipeline.device)
        try:
            pipeline.reload_text_encoder()
        except Exception:
            logger.exception("reload_text_encoder failed")
        ctx_list, neg = [], None
        try:
            for p in prompts:
                cv, nv = pipeline.encode_prompt(p or "", " ")
                ctx_list.append(cv.detach().to("cpu"))
                if neg is None and nv is not None:
                    neg = nv.detach().to("cpu")
        finally:
            try:
                pipeline.unload_text_encoder()
            except Exception:
                pass
            clean_memory_on_device(pipeline.device)
        # Invalidate the slider-preview prompt cache (TE was cycled).
        self._prompt_cache_key = None
        self._prompt_cache = None
        return ctx_list, neg

    @staticmethod
    def interp_waypoints(vecs, t, mode="lerp"):
        """Piecewise interpolation across an ordered list of conditioning tensors.
        `t` in [0,1] walks the whole chain (0 -> vecs[0], 1 -> vecs[-1]).

        mode:
          "lerp"  — plain linear interpolation (original). The blended embedding's
                    norm sags toward each segment midpoint, which under-conditions
                    the model and tends to dip brightness/contrast there.
          "norm"  — per-token norm-preserving lerp: same direction blend as lerp,
                    but each token's magnitude is rescaled to the interpolated
                    endpoint norm, so conditioning strength stays constant across
                    the sweep (flatter brightness, minimal change to the look).
          "slerp" — per-token spherical interpolation: constant angular velocity
                    along the arc, full-strength conditioning throughout. Smoothest
                    semantic glide; a bigger departure from plain lerp. Done per
                    token (norm over the feature dim) so the sequence structure is
                    preserved — unlike a whole-tensor slerp which rotates all tokens
                    through one shared angle and smears per-token alignment.

        Text embeddings aren't truly sphere-distributed, so "norm" is the low-risk
        tweak (keeps lerp's direction, only fixes the magnitude sag) and "slerp"
        the more aggressive option."""
        import torch
        if len(vecs) == 1:
            return vecs[0]
        t = min(max(float(t), 0.0), 1.0)
        segs = len(vecs) - 1
        pos = t * segs
        i = min(int(pos), segs - 1)
        local = pos - i
        a, b = vecs[i].float(), vecs[i + 1].float()
        if mode in (None, "lerp"):
            return torch.lerp(a, b, local).to(vecs[i].dtype)

        eps = 1e-6
        na = a.norm(dim=-1, keepdim=True)
        nb = b.norm(dim=-1, keepdim=True)
        target = (1.0 - local) * na + local * nb        # interpolated per-token norm

        if mode == "norm":
            out = torch.lerp(a, b, local)
            out = out * (target / out.norm(dim=-1, keepdim=True).clamp_min(eps))
            return out.to(vecs[i].dtype)

        if mode == "slerp":
            ua = a / na.clamp_min(eps)
            ub = b / nb.clamp_min(eps)
            dot = (ua * ub).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            omega = torch.acos(dot)
            so = torch.sin(omega)
            w_a = torch.sin((1.0 - local) * omega) / so
            w_b = torch.sin(local * omega) / so
            arc = w_a * ua + w_b * ub                    # unit per-token direction
            lin = torch.lerp(ua, ub, local)              # fallback for ~parallel tokens
            dir_ = torch.where(so < 1e-4, lin, arc)
            return (dir_ * target).to(vecs[i].dtype)

        return torch.lerp(a, b, local).to(vecs[i].dtype)

    def generate_preview(self, state: SliderState, seed_b=None, travel_t=None,
                         override_ctx=None, override_neg_ctx=None,
                         prev_latent=None, prev_latent_strength=1.0) -> Image.Image:
        """Apply state, run a 4-step Distilled generation, return PIL image.

        Seed travel: when `seed_b` is given, the initial noise is a spherical
        interpolation (slerp) between the noise of `state.seed` and `seed_b` at
        fraction `travel_t` (0=state.seed, 1=seed_b). Slerp walks the noise
        hypersphere so the composition flows continuously between the two seeds
        — lerp would collapse the norm mid-way and go mushy.

        Prompt travel: when `override_ctx` is given, it's used as the (raw,
        pre-prc_txt) conditioning instead of encoding state.prompt — pass an
        interpolated waypoint embedding here for a smooth meaning-morph on a
        fixed seed. `override_neg_ctx` overrides the negative conditioning.

        The activation cache is bypassed on either travel path (it's keyed on
        state.seed / state.prompt, which are constant across a sweep).

        Inlines the proven training-loop sample path (trainer.do_inference)
        instead of pipeline.generate() — the latter is a dormant untested
        method that produces blocky noise in this call site. See trainer.py
        sample_image_inference / do_inference for the reference path.
        """
        if self.pipeline is None or not self.pipeline.is_loaded:
            raise RuntimeError("Pipeline not loaded.")

        import os
        import numpy as np
        import torch
        from diffusers.utils.torch_utils import randn_tensor
        from fizgig.klein.model_utils import denoise, denoise_cfg, get_schedule, get_simple_euler_schedule
        from fizgig.klein.position import prc_img, prc_txt, scatter_ids
        from fizgig.utils.device import clean_memory_on_device

        # --- Diagnostic log for Distilled debugging ---
        log_lines = []
        def dlog(msg):
            logger.info("[REPAIR] %s", msg)
            log_lines.append(msg)

        self.apply_state(state)

        pipeline = self.pipeline
        device = pipeline.device
        defaults = pipeline.model_info.defaults
        num_steps = int(defaults.get("num_steps", 4))
        guidance_scale = float(defaults.get("guidance", 1.0))
        cfg_scale = float(defaults.get("cfg_scale", guidance_scale))

        dlog(f"=== generate_preview ===")
        dlog(f"model_version: {pipeline.model_info.architecture} distilled={pipeline.is_distilled}")
        dlog(f"defaults: {defaults}")
        dlog(f"num_steps={num_steps} guidance_scale={guidance_scale} cfg_scale={cfg_scale}")
        dlog(f"state.seed={state.seed} prompt={state.prompt!r}")
        dlog(f"state.preview_width={state.preview_width} preview_height={state.preview_height}")
        dlog(f"dit dtype: {pipeline.dit_dtype}")
        dlog(f"dit device: {next(pipeline.dit.parameters()).device}")
        # Sample a weight to check its dtype
        for n, p in pipeline.dit.named_parameters():
            dlog(f"sample DiT param: {n} dtype={p.dtype} shape={list(p.shape)} device={p.device}")
            break
        # LoRA status
        if self.primary_network is not None:
            n_enabled = sum(1 for m in self.primary_network.unet_loras if m.enabled)
            n_total = len(self.primary_network.unet_loras)
            mults = {m.multiplier for m in self.primary_network.unet_loras}
            dlog(f"primary LoRA: {n_enabled}/{n_total} modules enabled, unique multipliers={mults}")

        # Round to multiples of 16 (Klein patch size).
        width = (state.preview_width // 16) * 16
        height = (state.preview_height // 16) * 16

        # Encode prompt — cached to avoid reloading TE on slider-only changes.
        prompt = state.prompt or ""
        prompt_key = (prompt, self.primary_path)
        if override_ctx is not None:
            # Prompt travel: caller supplies a pre-interpolated conditioning.
            ctx_vec, neg_ctx_vec = override_ctx, override_neg_ctx
            dlog("Prompt travel: using supplied (interpolated) conditioning")
        elif self._prompt_cache_key == prompt_key and self._prompt_cache is not None:
            ctx_vec, neg_ctx_vec = self._prompt_cache
            dlog("Prompt cache hit — skipping TE reload")
        else:
            # Phase boundary 1: release allocator state before TE spike.
            clean_memory_on_device(device)
            try:
                pipeline.reload_text_encoder()
            except Exception:
                logger.exception("reload_text_encoder failed")
            ctx_vec, neg_ctx_vec = pipeline.encode_prompt(prompt, " ")
            try:
                pipeline.unload_text_encoder()
            except Exception:
                pass
            # Phase boundary 2: reclaim TE footprint before DiT forward.
            clean_memory_on_device(device)
            self._prompt_cache_key = prompt_key
            self._prompt_cache = (ctx_vec, neg_ctx_vec)
            dlog("Prompt encoded and cached")

        def _stats(t):
            tf = t.detach().float()
            return f"shape={list(t.shape)} dtype={t.dtype} device={t.device} mean={tf.mean().item():.4f} std={tf.std().item():.4f} min={tf.min().item():.4f} max={tf.max().item():.4f} norm={tf.norm().item():.2f}"

        dlog(f"ctx_vec raw: {_stats(ctx_vec)}")
        dlog(f"neg_ctx_vec raw: {_stats(neg_ctx_vec)}")

        ctx = ctx_vec.to(device=device, dtype=torch.bfloat16)
        ctx, ctx_ids = prc_txt(ctx)
        dlog(f"after prc_txt: ctx {_stats(ctx)} ctx_ids shape={list(ctx_ids.shape)} dtype={ctx_ids.dtype}")

        neg_ctx = neg_ctx_vec.to(device=device, dtype=torch.bfloat16) if neg_ctx_vec is not None else None
        neg_ctx_ids = None
        if neg_ctx is not None:
            neg_ctx, neg_ctx_ids = prc_txt(neg_ctx)

        # Seeded latent noise (slerp between two seeds when seed-travelling).
        packed_h, packed_w = height // 16, width // 16
        noise_shape = (1, 128, packed_h, packed_w)

        def _seed_noise(seed):
            g = torch.Generator(device=device)
            if seed is not None:
                g.manual_seed(int(seed))
            return randn_tensor(noise_shape, generator=g, device=device, dtype=torch.bfloat16)

        seed_travel = seed_b is not None
        bypass_cache = seed_travel or (override_ctx is not None)
        if seed_travel:
            latents = _slerp(float(travel_t or 0.0), _seed_noise(state.seed), _seed_noise(seed_b))
            dlog(f"seed travel: {state.seed} -> {seed_b} @ t={travel_t}")
        else:
            latents = _seed_noise(state.seed)
        x, x_ids = prc_img(latents)
        dlog(f"initial latent packed: {_stats(x)} x_ids shape={list(x_ids.shape)}")

        # Reference image (Klein edit conditioning) — encode once, inject into the
        # positive/cond forward only. Disables Turbo (the activation cache is keyed
        # without the ref sequence) and shifts no other behavior when absent.
        ref_tokens, ref_ids = self._build_ref_tokens(
            state, prev_latent=prev_latent, prev_latent_strength=prev_latent_strength)
        has_ref = ref_tokens is not None
        if has_ref:
            dlog(f"ref active: tokens={list(ref_tokens.shape)} "
                 f"strength={state.ref_strength} mp={state.ref_megapixels} "
                 f"path={os.path.basename(state.ref_image_path)}")

        # Prepare block-swap (no-op if not enabled).
        if hasattr(pipeline.dit, "_offloader") and pipeline.dit._offloader is not None:
            pipeline.dit._offloader.set_forward_only(True)
            pipeline.dit.prepare_block_swap_before_forward()

        # Match ComfyUI's Euler Simple schedule (shift=2.02, evenly-spaced)
        if pipeline.is_distilled:
            timesteps = get_simple_euler_schedule(num_steps)
        else:
            timesteps = get_schedule(num_steps, x.shape[1], None)
        dlog(f"timesteps ({len(timesteps)}): {[round(t, 4) for t in timesteps]}")

        pipeline.dit.eval()
        import torch as _torch

        # --- v2 Turbo: determine if activation cache is usable ---
        cache_key = (self.primary_path, self.donor_path, state.seed,
                     state.prompt, width, height,
                     state.ref_image_path, round(float(state.ref_megapixels), 4),
                     round(float(state.ref_strength), 4))
        # Resume point from a STATE DIFF against what the cache was built from — race-free, so
        # edits made while a previous render was in flight are never dropped.
        changed = (state.diff_blocks(self._act_cache_state)
                   if self._act_cache_state is not None else None)
        can_use_cache = (
            self._turbo_enabled
            and pipeline.is_distilled
            and not has_ref
            and not bypass_cache
            and self._act_cache is not None
            and self._act_cache_key == cache_key
            and changed
        )
        # No VRAM pre-check — the try/except around forward_cached() handles
        # OOM gracefully by falling back to full forward.

        resume_point = self._compute_resume_point(changed) if can_use_cache else None
        if resume_point:
            dlog(f"Turbo resume from {resume_point[0]}_{resume_point[1]}")
        elif self._turbo_enabled and pipeline.is_distilled:
            dlog("Turbo: populating cache (first run or invalidated)")

        from fizgig.klein.model import ActivationCacheEntry

        if pipeline.is_distilled:
            guidance_vec = _torch.full((x.shape[0],), guidance_scale, device=x.device, dtype=x.dtype)
            new_cache = {}
            turbo_fallback = False

            for si, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                t_vec = _torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)

                # Decide whether to use cached forward for this step
                use_cached = (
                    self._turbo_enabled
                    and not turbo_fallback
                    and not has_ref
                    and not bypass_cache
                    and (can_use_cache or (not can_use_cache and self._turbo_enabled))
                )

                if use_cached:
                    step_resume = resume_point if can_use_cache else None
                    step_cached = self._act_cache.get(si) if can_use_cache else None
                    new_entry = ActivationCacheEntry(
                        double_outputs=[None] * 8,
                        transition_img=None, transition_pe=None,
                        single_outputs=[None] * 24,
                    )
                    try:
                        if hasattr(pipeline.dit, 'prepare_block_swap_before_forward'):
                            pipeline.dit.prepare_block_swap_before_forward()
                        with _torch.no_grad(), _torch.autocast(device_type=x.device.type, dtype=x.dtype):
                            pred = pipeline.dit.forward_cached(
                                x=x, x_ids=x_ids, timesteps=t_vec,
                                ctx=ctx, ctx_ids=ctx_ids, guidance=guidance_vec,
                                resume_from=step_resume, cached=step_cached,
                                new_cache=new_entry,
                            )
                        new_cache[si] = new_entry
                    except Exception:
                        logger.warning("Turbo cache error at step %d; falling back to full forward", si, exc_info=True)
                        self._invalidate_activation_cache()
                        turbo_fallback = True
                        with _torch.no_grad(), _torch.autocast(device_type=x.device.type, dtype=x.dtype):
                            pred = pipeline.dit(x=x, x_ids=x_ids, timesteps=t_vec, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance_vec)
                else:
                    if has_ref:
                        x_fwd = _torch.cat((x, ref_tokens), dim=1)
                        x_fwd_ids = _torch.cat((x_ids, ref_ids), dim=1)
                    else:
                        x_fwd, x_fwd_ids = x, x_ids
                    with _torch.no_grad(), _torch.autocast(device_type=x.device.type, dtype=x.dtype):
                        pred = pipeline.dit(x=x_fwd, x_ids=x_fwd_ids, timesteps=t_vec, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance_vec)
                    if has_ref:
                        pred = pred[:, : x.shape[1]]

                dlog(f"step {si}: t_curr={t_curr:.4f} t_prev={t_prev:.4f} dt={(t_prev - t_curr):+.4f}")
                dlog(f"  pred: {_stats(pred)}")
                x = x + (t_prev - t_curr) * pred
                dlog(f"  x after step: {_stats(x)}")

            # Store cache for next preview (+ the state it reflects, for the resume diff)
            if self._turbo_enabled and not turbo_fallback and new_cache:
                self._act_cache = new_cache
                self._act_cache_key = cache_key
                self._act_cache_state = state.copy()
        elif has_ref:
            # Base + CFG with a reference: the ref conditions the POSITIVE pass
            # only (uncond stays ref-free), mirroring ComfyUI's ReferenceLatent.
            # Inlined here rather than via denoise_cfg, which applies the ref to
            # both passes and is shared with the trainer sample path.
            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                t_vec = _torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)
                x_cond = _torch.cat((x, ref_tokens), dim=1)
                x_cond_ids = _torch.cat((x_ids, ref_ids), dim=1)
                with _torch.no_grad(), _torch.autocast(device_type=x.device.type, dtype=x.dtype):
                    pred_cond = pipeline.dit(x=x_cond, x_ids=x_cond_ids, timesteps=t_vec,
                                             ctx=ctx, ctx_ids=ctx_ids, guidance=None)
                    pred_uncond = pipeline.dit(x=x, x_ids=x_ids, timesteps=t_vec,
                                               ctx=neg_ctx, ctx_ids=neg_ctx_ids, guidance=None)
                pred_cond = pred_cond[:, : x.shape[1]]
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                x = x + (t_prev - t_curr) * pred
            dlog(f"after ref CFG denoise: x {_stats(x)}")
        else:
            x = denoise_cfg(
                pipeline.dit, x, x_ids, ctx, ctx_ids, neg_ctx, neg_ctx_ids,
                timesteps=timesteps, guidance=cfg_scale,
            )
            dlog(f"after denoise_cfg: x {_stats(x)}")

        # Unpack latent → VAE decode.
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        dlog(f"unpacked latent: {_stats(x)}")
        latent = x.to(pipeline.vae.dtype)
        del x

        # Cache the clean latent ([128, H/16, W/16] — same space as vae.encode) so
        # sequential travel can reuse it as the NEXT frame's reference latent directly,
        # skipping the lossy decode→PNG→encode round trip that otherwise compounds each
        # frame (the cause of quality loss at high sequential-reference strength).
        self._last_frame_latent = latent[0].detach().clone()

        # Phase boundary 3: release DiT activation scratch before VAE claims GPU.
        clean_memory_on_device(device)

        pipeline.vae.to(device)
        pipeline.vae.eval()
        with torch.no_grad():
            pixels = pipeline.vae.decode(latent)
        dlog(f"vae decoded: {_stats(pixels)}")
        del latent
        pixels = pixels.to(torch.float32).cpu()
        pixels = (pixels / 2 + 0.5).clamp(0, 1)
        pipeline.vae.to("cpu")
        # Phase boundary 4: release VAE decode scratch before handing PIL back.
        clean_memory_on_device(device)

        img_np = (pixels[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        dlog(f"img_np stats: shape={img_np.shape} dtype={img_np.dtype} mean={img_np.mean():.2f} std={img_np.std():.2f}")
        img = Image.fromarray(img_np)

        # Dump log to a file users can share.
        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            log_path = os.path.join(repo_root, "repair_studio_last_preview.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(log_lines))
            logger.info("Wrote diagnostic log to %s", log_path)
        except Exception:
            logger.exception("Failed to write diagnostic log")

        return img

    def generate_baseline(self, state: SliderState) -> Image.Image:
        """Baseline = primary at default 1.0 / all enabled, donor fully off.

        Cached on (primary_path, seed, prompt, w, h) — slider tweaks don't
        invalidate it. Cache cleared on primary swap or reset.
        """
        key = (self.primary_path, state.seed, state.prompt,
               state.preview_width, state.preview_height,
               state.ref_image_path, round(float(state.ref_megapixels), 4),
               round(float(state.ref_strength), 4))
        if self._baseline_cache_key == key and self._baseline_cache_image is not None:
            return self._baseline_cache_image

        baseline_state = SliderState.default_klein9b()
        baseline_state.seed = state.seed
        baseline_state.prompt = state.prompt
        baseline_state.preview_width = state.preview_width
        baseline_state.preview_height = state.preview_height
        # Condition the baseline on the same reference, so the side-by-side
        # differs only by the slider tweaks (not by ref presence).
        baseline_state.ref_image_path = state.ref_image_path
        baseline_state.ref_megapixels = state.ref_megapixels
        baseline_state.ref_strength = state.ref_strength

        img = self.generate_preview(baseline_state)
        self._baseline_cache_key = key
        self._baseline_cache_image = img
        return img

    def _invalidate_baseline_cache(self) -> None:
        self._baseline_cache_key = None
        self._baseline_cache_image = None
