"""Repair-Studio engine for Krea 2 — the parallel of `engine.RepairEngine`.

Klein's RepairEngine inlines a 270-line Klein denoise loop that doesn't transfer; Krea 2's
`sampling.sample` is a complete sampler, so the Krea 2 preview is simpler: apply the slider
state to the LoRA networks (the model-agnostic `set_module_*_by_pattern` API) and call
`sampling.sample`. Previews always render on the fp8 Turbo (8-step, CFG-free). The per-block
slider config is the shared `SliderState`; only the block ids/regex are Krea 2-specific
(`krea2_blocks`).

Public surface mirrors RepairEngine so the Repair Studio UI can drive either: `ensure_pipeline`,
`load_primary`, `load_donor` / `unload_donor`, `apply_state`, `generate_preview`, `reset`, and
the `primary_network` / `donor_network` / `*_block_ids` / `*_path` / `primary_hash` attributes,
plus a `pipeline` holder exposing `is_loaded`.
"""

import gc
import logging
import os
import threading
from typing import Optional, Set

import torch
from PIL import Image

from fizgig.repair_studio.krea2_blocks import block_regex_krea2, extract_block_ids_krea2

logger = logging.getLogger(__name__)


def _slerp(t, a, b):
    """Spherical linear interpolation between two noise tensors at fraction t (0 -> a, 1 -> b).
    Walks the hypersphere so the interpolated noise keeps a constant norm — lerp would shrink it
    mid-way and the sample would go mushy. Falls back to lerp for (anti)parallel endpoints.
    (Local copy of RepairEngine._slerp so krea2_engine doesn't import the heavy Klein engine.)"""
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


class _Loaded:
    """Tiny stand-in for KleinInferencePipeline.is_loaded so the shared UI checks
    `engine.pipeline.is_loaded` uniformly across both engines."""

    def __init__(self):
        self.is_loaded = True


def _apply_lora(target, sd, multiplier, device, dtype):
    """Normalize foreign formats, build the network, apply it live, load the weights. Returns
    the network. Mirrors the verified Context-LoRA / preview path (ensure_kohya -> apply_to ->
    load_state_dict — create_network_from_weights only builds STRUCTURE, the values must be
    loaded or lora_up stays 0)."""
    from fizgig.networks.lora import create_network_from_weights, ensure_kohya_lora_state_dict
    sd = ensure_kohya_lora_state_dict(sd)
    net = create_network_from_weights(None, float(multiplier), sd, None, target, for_inference=True)
    net.apply_to(text_encoders=None, unet=target, apply_text_encoder=False, apply_unet=True)
    net.load_state_dict(sd, strict=False)
    net.to(device=device, dtype=dtype).eval()
    return net


class Krea2RepairEngine:
    def __init__(self):
        self.pipeline: Optional[_Loaded] = None
        self.turbo = None          # fp8 Turbo DiT
        self.ae = None             # Qwen-Image VAE (kept on CPU; sampling moves it for decode)
        self.te_path: Optional[str] = None
        self.device = "cuda"
        self.dtype = torch.bfloat16

        self.primary_network = None
        self.donor_network = None
        self.primary_path: Optional[str] = None
        self.donor_path: Optional[str] = None
        self.primary_block_ids: Set[str] = set()
        self.donor_block_ids: Set[str] = set()
        self.primary_hash: Optional[str] = None

        # Encoded-prompt cache: prompt -> (txt, txtmask) on CPU. Slider tweaks don't change the
        # prompt, so the 8 GB TE only loads when the prompt actually changes.
        self._prompt_cache_key: Optional[str] = None
        self._prompt_cache = None
        # Cooperative cancellation: set to abort an in-flight render so a new edit restarts it
        # immediately instead of queueing behind the full 8-step pass.
        self._cancel_event = threading.Event()
        # Optional progress hook: called (step_done, total_steps) once per denoising step,
        # from the render thread. The GUI sets it to drive a determinate progress bar.
        self.on_step = None
        # Baseline render cache (LoRA at original strengths; slider tweaks don't invalidate it).
        self._baseline_cache_key = None
        self._baseline_cache_image: Optional[Image.Image] = None
        # Klein's sequential-travel latent chain has no Krea 2 analogue; kept as a None attribute
        # so LoRA Royale's shared workers (which read eng._last_frame_latent) don't AttributeError.
        self._last_frame_latent = None

        # Turbo Preview — per-step activation cache (forward_cached). Tweaking a late block
        # skips the earlier blocks. Cache key: (primary, donor, seed, prompt, w, h). The resume
        # point is computed by DIFFING the render's state against `_act_cache_state` (the state
        # the cache was built from) — not a mutable changed-set — so rapid edits during an
        # in-flight render are never lost.
        self._turbo_enabled = True
        self._act_cache = None          # {step_idx: KreaActivationCacheEntry}
        self._act_cache_key = None
        self._act_cache_state = None    # SliderState the current cache reflects

    # ----- pipeline + LoRA loading -------------------------------------------
    def ensure_pipeline(self, turbo_path: str, vae_path: str, text_encoder_path: str,
                        device: str = "cuda", model_kind: str = "turbo",
                        blocks_to_swap: int = 0, int8: bool = False, **_ignored) -> None:
        """Load the preview DiT + VAE once (TE loads on demand per prompt-encode).

        model_kind: 'turbo' (fp8, 8-step, CFG-free — the default) or 'raw' (the undistilled
        base — slower, more steps). The DiT loader auto-detects pre-quant fp8 either way.

        blocks_to_swap: forward-only block swap on the Turbo so the workbench fits smaller cards
        (mirrors Klein's inference block swap). The SingleStreamDiT forward already streams blocks
        via the offloader; we just load on CPU, place the resident blocks, and switch to the
        forward-only (inference) offload mode."""
        if self.pipeline is not None and self.pipeline.is_loaded:
            return
        from fizgig.krea2.utils import load_krea2_dit
        from fizgig.krea2.vae_loader import load_vae
        self.device = device
        self.te_path = text_encoder_path
        self.model_kind = model_kind
        self._default_steps = 20 if model_kind == "raw" else 8
        loading_device = "cpu" if blocks_to_swap > 0 else device
        self.turbo = load_krea2_dit(turbo_path, device=device, dtype=self.dtype,
                                    loading_device=loading_device)
        if int8:
            # INT8 (W8A8) fast inference — quantize the block Linears BEFORE block swap so the
            # offloader stages int8 (see modules/int8.py). Quantize on the model's load device so a
            # swapped (CPU-loaded) model doesn't briefly need the whole int8 model resident on GPU.
            from fizgig.modules.int8 import apply_int8_quantization
            from fizgig.krea2.utils import (KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
                                            KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS)
            apply_int8_quantization(self.turbo, target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
                                    exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS,
                                    compute_device=torch.device(loading_device))
        if blocks_to_swap > 0:
            from fizgig.krea2.offloading import BlockSwapConfig
            self.turbo.enable_block_swap(blocks_to_swap, BlockSwapConfig(torch.device(device), supports_backward=False))
            self.turbo.move_to_device_except_swap_blocks(torch.device(device))
            self.turbo.switch_block_swap_for_inference()
        self.ae = load_vae(vae_path, input_channels=3, device="cpu", disable_mmap=True)
        self.turbo.eval()
        self.pipeline = _Loaded()
        logger.info("Krea2 Repair engine ready (%s=%s, block_swap=%d)", model_kind, turbo_path, blocks_to_swap)

    def load_primary(self, path: str) -> None:
        if self.pipeline is None or not self.pipeline.is_loaded:
            raise RuntimeError("Pipeline not loaded; call ensure_pipeline() first.")
        if self.primary_network is not None:
            raise RuntimeError("Primary already loaded — call reset() to swap.")
        from safetensors.torch import load_file
        self.primary_network = _apply_lora(self.turbo, load_file(path), 1.0, self.device, self.dtype)
        self.primary_path = path
        self.primary_block_ids = extract_block_ids_krea2(self.primary_network)
        self._invalidate_baseline_cache()
        try:
            from fizgig.profiler.visualize import compute_lora_hash
            self.primary_hash = compute_lora_hash(path)
        except Exception:
            self.primary_hash = None
        logger.info("Krea2 primary loaded: %s (%d blocks)", path, len(self.primary_block_ids))

    def swap_primary_weights(self, path: str) -> bool:
        """LoRA Royale fast path: swap the primary network's weights to another checkpoint of the
        SAME structure (epochs of one run) WITHOUT reloading the Turbo or re-patching the DiT —
        load_state_dict in place. Returns True on a clean swap, False if the structure doesn't
        match (different rank / unrelated LoRA), in which case the caller should reset() +
        load_primary(). Model-agnostic mirror of RepairEngine.swap_primary_weights."""
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
            logger.info("swap_primary_weights: structure/shape mismatch, needs full reload (%s)", e)
            return False
        if sd and len(info.unexpected_keys) > 0.5 * len(sd):
            return False
        self.primary_network.to(device=self.device, dtype=self.dtype)
        self.primary_path = path
        self.primary_block_ids = extract_block_ids_krea2(self.primary_network)
        try:
            from fizgig.profiler.visualize import compute_lora_hash
            self.primary_hash = compute_lora_hash(path)
        except Exception:
            self.primary_hash = None
        self._invalidate_baseline_cache()
        self._invalidate_activation_cache()
        return True

    def load_donor(self, path: str) -> None:
        if self.primary_network is None:
            raise RuntimeError("Load primary LoRA before donor.")
        if self.donor_network is not None:
            raise RuntimeError("Donor already loaded — unload_donor() or reset() first.")
        from safetensors.torch import load_file
        net = _apply_lora(self.turbo, load_file(path), 1.0, self.device, self.dtype)
        net.set_enabled(False)  # donor blocks are opt-in per-slider
        self.donor_network = net
        self.donor_path = path
        self.donor_block_ids = extract_block_ids_krea2(net)
        logger.info("Krea2 donor loaded: %s (%d blocks)", path, len(self.donor_block_ids))

    def unload_donor(self) -> None:
        if self.donor_network is not None:
            self.donor_network.set_enabled(False)
            self.donor_network = None
            self.donor_path = None
            self.donor_block_ids = set()

    # ----- slider state ------------------------------------------------------
    def apply_state(self, state) -> None:
        """Push the per-block slider config into the live networks (regex-based, no reload)."""
        if self.primary_network is None:
            return
        for bid, bs in state.blocks.items():
            try:
                pat = block_regex_krea2(bid)
            except ValueError:
                continue
            self.primary_network.set_module_enabled_by_pattern(pat, bool(bs.primary_enabled))
            self.primary_network.set_module_multiplier_by_pattern(pat, float(bs.primary_strength))
            if self.donor_network is not None:
                self.donor_network.set_module_enabled_by_pattern(pat, bool(bs.donor_enabled))
                self.donor_network.set_module_multiplier_by_pattern(pat, float(bs.donor_strength))

    # ----- cancellation ------------------------------------------------------
    def request_cancel(self) -> None:
        """Signal the in-flight render to abort at the next step boundary."""
        self._cancel_event.set()

    def clear_cancel(self) -> None:
        self._cancel_event.clear()

    # ----- preview -----------------------------------------------------------
    def _encode_prompt(self, prompt: str, ref_path: str = "", ref_megapixels: float = 1.0):
        """Encode the prompt once (load TE -> encode -> free), cached on (prompt, ref, mp).

        An optional reference image is pushed through Qwen3-VL's vision path so the conditioning
        becomes visually aware of it (Krea 2's reference mechanism — see embedder.py). Re-encodes
        only when the prompt or reference changes (slider tweaks don't)."""
        key = (prompt, ref_path or "", round(float(ref_megapixels), 4))
        if self._prompt_cache_key == key and self._prompt_cache is not None:
            return self._prompt_cache
        from fizgig.krea2.utils import load_krea2_text_encoder
        from fizgig.krea2 import sampling
        images = None
        if ref_path and os.path.exists(ref_path):
            from PIL import Image as _Image
            images = [[_Image.open(ref_path)]]
        enc = load_krea2_text_encoder(self.te_path, dtype=self.dtype, device=self.device)
        txt, txtmask, _, _ = sampling.encode_prompts(enc, [prompt], cfg=False,
                                                     images=images, vision_megapixels=float(ref_megapixels))
        del enc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._prompt_cache_key = key
        self._prompt_cache = (txt.cpu(), txtmask.cpu())
        return self._prompt_cache

    def generate_preview(self, state, *, seed: Optional[int] = None,
                         prompt: Optional[str] = None, width: Optional[int] = None,
                         height: Optional[int] = None, steps: Optional[int] = None,
                         seed_b: Optional[int] = None, travel_t: float = 0.0,
                         override_ctx=None, override_neg_ctx=None,
                         prev_latent=None, prev_latent_strength: float = 1.0) -> Image.Image:
        """Apply the slider state and render one preview with the live LoRA(s). Always a full
        forward — the per-step activation cache (forward_cached) is NOT used here.

        Why: across a multi-step denoise, img is chained (img_{t+1} = img_t + dt*model(img_t)).
        A block tweak changes step 0's output, which changes step 1's input img, which makes the
        cached prefix of EVERY later step stale. Reusing it (Klein's "approximate" cache) is
        visually fine at Klein's 4 steps but compounds badly over krea's 8 — the symptom was
        tweaks only partially registering. forward_cached stays on the model (correct within a
        single pass, and a future multi-step-aware cache could use it), but the live preview does
        the correct full forward, which on the 8-step Turbo is already fast (~7 s).

        LoRA Royale travel hooks (model-agnostic, no Klein reference-latent involved):
        - seed_b/travel_t: initial noise is slerp(noise(seed), noise(seed_b), travel_t) — a smooth
          seed morph for seed-travel. travel_t=0 -> seed, 1 -> seed_b.
        - override_ctx: a precomputed (txt, txtmask) pair (from interp_waypoints) used instead of
          encoding state.prompt — drives prompt-travel. The Krea 2 reference is the vision-path
          image on the state (no latent chaining), applied via the normal _encode_prompt path.

        override_neg_ctx / prev_latent / prev_latent_strength are accepted for call-signature
        parity with RepairEngine (so LoRA Royale's shared workers drive either engine) but are
        Klein-only: the Turbo is CFG-free (no negative) and Krea 2 has no reference-latent, so the
        morph comes purely from the seed slerp / prompt interpolation, with the vision-path image
        as the (optional) per-frame anchor. They're intentionally ignored here."""
        self.apply_state(state)
        prompt = prompt if prompt is not None else state.prompt
        seed = seed if seed is not None else state.seed
        width = width or state.preview_width
        height = height or state.preview_height
        steps = steps or getattr(self, "_default_steps", 8)
        # Reference image (if any) goes through the Qwen3-VL vision path. ref_strength is a Klein
        # edit-conditioning knob with no Krea 2 analogue, so it's not used here; ref_megapixels
        # maps to the vision encoder's downscale cap.
        if override_ctx is not None:
            txt, txtmask = override_ctx
        else:
            txt, txtmask = self._encode_prompt(prompt, getattr(state, "ref_image_path", ""),
                                               getattr(state, "ref_megapixels", 1.0))
        txt = txt.to(self.device, dtype=self.dtype)
        txtmask = txtmask.to(self.device)

        noise = None
        if seed_b is not None:
            noise = _slerp(float(travel_t or 0.0),
                           self._seed_noise(seed, width, height),
                           self._seed_noise(seed_b, width, height))

        from fizgig.krea2 import sampling
        # should_abort is polled once per step, so counting its calls IS the step counter —
        # the sampler itself has no progress callback to thread through.
        _done = [0]

        def _abort_and_count():
            cb = self.on_step
            if cb is not None:
                try:
                    cb(_done[0], steps)
                except Exception:
                    pass
            _done[0] += 1
            return self._cancel_event.is_set()

        with torch.no_grad():
            imgs = sampling.sample(self.turbo, self.ae, txt, txtmask, untxt=None, untxtmask=None,
                                   device=self.device, dtype=self.dtype, width=width, height=height,
                                   steps=steps, cfg_scale=1.0, mu=1.15, seed=seed,
                                   should_abort=_abort_and_count, noise=noise)
        return imgs[0]

    def _seed_noise(self, seed, width, height):
        """One seeded gaussian latent matching sampling.sample's geometry (n=1), for seed-travel
        slerp. Same channel count / compression / alignment the sampler would use."""
        from fizgig.krea2.sampling import roundup
        patch = self.turbo.config.patch
        compression = 2 ** len(self.ae.temperal_downsample)
        channels = self.ae.z_dim
        align = compression * patch
        width = roundup(width, align, "width")
        height = roundup(height, align, "height")
        return torch.randn(1, channels, height // compression, width // compression,
                           device=self.device, dtype=self.dtype,
                           generator=torch.Generator(device=self.device).manual_seed(int(seed)))

    # ----- prompt travel ------------------------------------------------------
    def encode_travel_prompts(self, prompts):
        """Encode waypoint prompts for prompt-travel — ALL in one batch so gather_valid_text pads
        them to a common length, making them shape-compatible for interp_waypoints. Returns a list
        of (txt, txtmask) CPU pairs (one per waypoint). The Turbo is CFG-free, so the negative is
        None — returned as (waypoints, None) to match RepairEngine.encode_travel_prompts' signature
        so LoRA Royale's shared prompt-travel worker can unpack either engine's result."""
        from fizgig.krea2.utils import load_krea2_text_encoder
        from fizgig.krea2 import sampling
        enc = load_krea2_text_encoder(self.te_path, dtype=self.dtype, device=self.device)
        txt, txtmask, _, _ = sampling.encode_prompts(enc, list(prompts), cfg=False)
        del enc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # The slider-preview prompt cache is stale (the TE was cycled).
        self._prompt_cache_key = None
        self._prompt_cache = None
        waypoints = [(txt[i:i + 1].cpu(), txtmask[i:i + 1].cpu()) for i in range(txt.shape[0])]
        return waypoints, None

    @staticmethod
    def interp_waypoints(vecs, t, mode="lerp"):
        """Piecewise interpolation across ordered (txt, txtmask) waypoints. `t` in [0,1] walks the
        whole chain (0 -> vecs[0], 1 -> vecs[-1]). Mirrors RepairEngine.interp_waypoints' lerp /
        norm / slerp on the embedding; the mask is the union of the two segment endpoints (a token
        attended in either neighbour stays attended). Returns (txt, txtmask)."""
        if len(vecs) == 1:
            return vecs[0]
        t = min(max(float(t), 0.0), 1.0)
        segs = len(vecs) - 1
        pos = t * segs
        i = min(int(pos), segs - 1)
        local = pos - i
        (ta, ma), (tb, mb) = vecs[i], vecs[i + 1]
        a, b = ta.float(), tb.float()
        eps = 1e-6
        if mode in (None, "lerp"):
            blended = torch.lerp(a, b, local)
        else:
            na = a.norm(dim=-1, keepdim=True)
            nb = b.norm(dim=-1, keepdim=True)
            target = (1.0 - local) * na + local * nb
            if mode == "norm":
                out = torch.lerp(a, b, local)
                blended = out * (target / out.norm(dim=-1, keepdim=True).clamp_min(eps))
            elif mode == "slerp":
                ua = a / na.clamp_min(eps)
                ub = b / nb.clamp_min(eps)
                dot = (ua * ub).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                omega = torch.acos(dot)
                so = torch.sin(omega)
                w_a = torch.sin((1.0 - local) * omega) / so
                w_b = torch.sin(local * omega) / so
                arc = w_a * ua + w_b * ub
                lin = torch.lerp(ua, ub, local)
                dir_ = torch.where(so < 1e-4, lin, arc)
                blended = dir_ * target
            else:
                blended = torch.lerp(a, b, local)
        return blended.to(ta.dtype), (ma.bool() | mb.bool())

    def _sample_cached(self, txt, txtmask, *, seed, width, height, steps, resume_from):
        """Denoise loop mirroring sampling.sample but via forward_cached, threading a per-step
        activation cache. Turbo preview is CFG-free (cfg_scale=1.0), so only the cond path runs."""
        from einops import rearrange
        from fizgig.krea2.sampling import prepare, timesteps, roundup
        from fizgig.krea2.model import KreaActivationCacheEntry
        model, ae, device, dtype = self.turbo, self.ae, self.device, self.dtype
        patch = model.config.patch
        compression = 2 ** len(ae.temperal_downsample)
        channels = ae.z_dim
        align = compression * patch
        width, height = roundup(width, align, "width"), roundup(height, align, "height")
        noise = torch.randn(1, channels, height // compression, width // compression,
                            device=device, dtype=dtype,
                            generator=torch.Generator(device=device).manual_seed(seed))
        img, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)
        x1 = (256 // align) ** 2
        x2 = (1280 // align) ** 2
        ts = timesteps(img.shape[1], steps, x1, x2, y1=0.5, y2=1.15, mu=1.15)
        new_cache = {}
        prev = self._act_cache or {}
        use_resume = resume_from is not None and bool(prev)
        with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
            for si, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
                t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
                entry = KreaActivationCacheEntry(block_inputs=[])
                step_cached = prev.get(si) if use_resume else None
                step_resume = resume_from if step_cached is not None else None
                v = model.forward_cached(img=img, context=txt, t=t, pos=pos, mask=mask,
                                         resume_from=step_resume, cached=step_cached, new_cache=entry)
                new_cache[si] = entry
                img = img + (tprev - tcurr) * v
        img = rearrange(img, "b (h w) (c ph pw) -> b c 1 (h ph) (w pw)",
                        ph=patch, pw=patch, h=height // align, w=width // align)
        ae = ae.to(img.device)
        pixels = ae.decode_to_pixels(img.to(torch.bfloat16))
        self.ae = ae.to("cpu")
        pixels = rearrange(pixels * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
        return Image.fromarray(pixels[0]), new_cache

    def generate_baseline(self, state) -> Image.Image:
        """Baseline = primary at default 1.0 / all enabled, donor off. Cached on
        (primary_path, seed, prompt, w, h) — slider tweaks don't invalidate it."""
        from fizgig.repair_studio.state import SliderState
        key = (self.primary_path, state.seed, state.prompt,
               state.preview_width, state.preview_height,
               getattr(state, "ref_image_path", ""), round(float(getattr(state, "ref_megapixels", 1.0)), 4))
        if self._baseline_cache_key == key and self._baseline_cache_image is not None:
            return self._baseline_cache_image
        base = SliderState.default_krea2()
        base.seed = state.seed
        base.prompt = state.prompt
        base.preview_width = state.preview_width
        base.preview_height = state.preview_height
        # Same reference as the tweaked side, so the comparison differs only by the slider tweaks.
        base.ref_image_path = getattr(state, "ref_image_path", "")
        base.ref_megapixels = getattr(state, "ref_megapixels", 1.0)
        img = self.generate_preview(base)
        self._baseline_cache_key = key
        self._baseline_cache_image = img
        return img

    def _invalidate_baseline_cache(self) -> None:
        self._baseline_cache_key = None
        self._baseline_cache_image = None

    def _invalidate_activation_cache(self) -> None:
        self._act_cache = None
        self._act_cache_key = None
        self._act_cache_state = None

    def mark_blocks_changed(self, blocks) -> None:
        # The GUI calls this on every slider change, but the resume point is derived from a
        # state diff at render time (race-free), so this is just a no-op compatibility shim.
        pass

    @staticmethod
    def _block_index(block_id):
        return int(block_id.split("_")[1]) if str(block_id).startswith("block_") else None

    def _resume_from_diff(self, state):
        """Earliest main-block index whose primary/donor differs from the cached state, or None
        (full recompute) if a non-main block (txtfusion) changed or there's no cached state.
        Diffing the FULL state against what the cache holds guarantees no edit is ever missed,
        even ones made while a previous render was in flight."""
        if self._act_cache_state is None:
            return None
        changed = state.diff_blocks(self._act_cache_state)
        if not changed:
            return None
        idxs = [self._block_index(b) for b in changed]
        if any(i is None for i in idxs):  # a txtfusion / non-main block changed -> full pass
            return None
        return min(idxs)

    # ----- teardown ----------------------------------------------------------
    def reset(self) -> None:
        """Full unload — drop networks (break forward-hook ref cycles) then the Turbo + VAE."""
        from fizgig.utils.device import release_module_tensors as _strip
        for net in (self.primary_network, self.donor_network):
            if net is not None:
                try:
                    _strip(net)     # husk the LoRA weights too — see the H3 reset comment
                    for lora in net.unet_loras:
                        lora.org_forward = None
                    net.unet_loras.clear()
                except Exception:
                    pass
        self.primary_network = None
        self.donor_network = None
        # Same defense as the H3 engine: if anything outside the engine pins the module
        # graph, dropping our reference leaks the VRAM — strip the storages first.
        from fizgig.utils.device import release_module_tensors
        release_module_tensors(self.turbo)
        release_module_tensors(self.ae)
        self.turbo = None
        self.ae = None
        self.pipeline = None
        self.primary_path = None
        self.donor_path = None
        self.primary_block_ids = set()
        self.donor_block_ids = set()
        self.primary_hash = None
        self._prompt_cache_key = None
        self._prompt_cache = None
        self._baseline_cache_key = None
        self._baseline_cache_image = None
        self._act_cache = None
        self._act_cache_key = None
        self._act_cache_state = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from fizgig.utils.device import report_cuda_leak, flush_reserved_vram
        report_cuda_leak("krea2-repair-reset")
        flush_reserved_vram("krea2-repair-reset")
