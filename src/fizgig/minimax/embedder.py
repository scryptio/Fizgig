"""MiniMax H3 text encoder: Qwen3-VL-32B language model (truncated to 50 layers).

The DiT is conditioned on the **unnormalized hidden state after language layer 50** — the
last-layer output of the truncated checkpoint, with NO final norm (comfy applies none). We
build a transformers Qwen3Model (its layout matches the checkpoint's `model.layers.*` 1:1),
strip the `model.` prefix, skip the vision tower (`visual.*` — unused for text-only captions),
NF4-quantize the big linears so the ~32B fits a 24-32 GB card, and replace the final norm with
Identity so last_hidden_state IS the raw layer-50 output.

Image LoRA training uses text-only captions (no vision blocks), tokenized raw with NO special
tokens — the H3 convention. The tokenizer is the Qwen3-VL one Fizgig already bundles
(src/fizgig/assets/qwen3vl_tokenizer — shared vocab across the 4B/32B sizes).

Output: [1, L, 5120] bf16, exactly what the DiT's condition_proj expects.
"""

import os

import torch
import torch.nn as nn

# The official safetensors safe_open(device="cpu") + get_tensor path memory-maps the file and
# slices the torch storage per tensor — on Windows, reading a 48 GB file that way hard-crashes
# (access violation, exit 0xC0000005) deep in torch.storage.__getitem__. The repo's own
# MemoryEfficientSafeOpen reads each tensor with a plain np.fromfile (no torch mmap-view), which
# is exactly why every large-model loader (krea2 / klein) uses it. Use it here too.
from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen

# Derived from the checkpoint tensor shapes (U8 weights are 4-bit-packed: real in-dim = 2x).
_QWEN3_32B_TRUNC50 = dict(
    hidden_size=5120, num_hidden_layers=50, num_attention_heads=64, num_key_value_heads=8,
    head_dim=128, intermediate_size=25600, vocab_size=151936, max_position_embeddings=262144,
    rms_norm_eps=1e-6, rope_theta=5000000.0, attention_bias=False, tie_word_embeddings=False,
)

# The Qwen3 decoder Linears to NF4 (the matmul bulk). embed_tokens stays as-is (int8 in the
# checkpoint / bf16 after load); the tiny q/k norms + layernorms stay bf16.
_NF4_SUFFIXES = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                 ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
                 ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight")


# ---------------------------------------------------------------------------
# ComfyUI "comfy_quant" dequant (nvfp4 + int8_tensorwise) — lets the small nvfp4-awq TE
# (~15 GB) stand in for the 48 GB bf16 file. We dequantize each weight back to bf16 per tensor
# at load, then NF4-quantize it on GPU exactly like the bf16 path, so nothing downstream changes.
#
# nvfp4 (comfy/float.py): W = fp4_e2m1 * block_scale * global_scale, block size 16 along the input
# dim. Stored as: weight U8 [out, in/2] (2 E2M1 values/byte, HIGH nibble = first/even element),
# weight_scale FP8-E4M3 [out, in/16] in ComfyUI's cuBLAS `to_blocked` swizzle (128x4 tiles ->
# (-1,32,16); see comfy/float.py::to_blocked — must be inverted to row-major before use),
# weight_scale_2 F32 scalar. AWQ layers (o_proj/down_proj) also carry pre_quant_scale [in]: comfy
# applies it as `input = input * pre_quant_scale` (ops.py), i.e. y=(x*s)@W^T = x@(W*s)^T — so we
# fold it into the weight columns by MULTIPLYING. For the other Linears (q/k/v, gate/up) the AWQ
# scale was folded into the PRECEDING norm weight at quantization time, so the checkpoint is
# self-consistent: load its norm weights as-is and no unfold is needed (validated per-column
# against the bf16 file: ratio W_dq/W_ref == ln_ref/ln_nvfp4 to <1%, all layer shapes).
# int8_tensorwise: W = int8 * weight_scale (per-tensor scalar).
# ---------------------------------------------------------------------------
# E2M1 magnitude for the low 3 bits (sign is bit 3): {0, .5, 1, 1.5, 2, 3, 4, 6}.
_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
# The same table extended over the full nibble, sign bit included — so decoding is a single
# gather rather than a magnitude lookup plus a separate sign tensor (see _nvfp4_dequant).
_E2M1_SIGNED = torch.cat([_E2M1_MAG, -_E2M1_MAG])


def _from_blocked(blocked, rows, cols):
    """Invert ComfyUI's to_blocked (comfy/float.py): (-1, 32, 16) tiles -> row-major [rows, cols].
    Verified as an exact roundtrip of to_blocked for the shapes in this checkpoint."""
    nrb = -(-rows // 128)
    ncb = -(-cols // 4)
    x = blocked.reshape(-1, 32, 4, 4).transpose(1, 2)
    x = x.reshape(nrb, ncb, 128, 4).permute(0, 2, 1, 3).reshape(nrb * 128, ncb * 4)
    return x[:rows, :cols].contiguous()


def _nvfp4_dequant(packed, block_scale_fp8, global_scale):
    """packed U8 [out, in/2] -> bf16 [out, in].  W = e2m1(code) * block_scale * global_scale.

    Written for CPU throughput — this runs once per Linear across ~350 layers of a 32B model,
    so the temporaries dominate. Two things keep it lean vs the obvious formulation:
      * ONE gather through a 16-entry SIGNED table, instead of a magnitude gather plus a
        separate sign tensor (the sign bit is just the top bit of the nibble, so it can live
        in the table);
      * the block scale is BROADCAST over each group of 16, instead of repeat_interleave'd to
        full width — which never materialises an [out, in] float32 copy of the scales.
    Measured bit-identical to the naive version, ~3x faster (209 -> 70 ms on a [5120, 8192]
    layer, i.e. ~73s -> ~25s across the whole encoder)."""
    out, in2 = packed.shape
    inp = in2 * 2
    dev = packed.device
    # device follows `packed`: this began as a load-time CPU helper, but Nvfp4Linear now calls it
    # inside a GPU forward, and an un-placed scratch tensor silently lands on CPU.
    bs = _from_blocked(block_scale_fp8.to(torch.float32).reshape(-1, 32, 16), out, inp // 16)
    gs = global_scale.to(torch.float32)
    table = _E2M1_SIGNED.to(dev)

    # ROW-CHUNKED, because the whole-matrix formulation transiently held an int64 gather
    # index (8 B/element — over a GB on the big layers), the fp32 values, and two fp32
    # products all at once: ~1.7 GB per forward, which is exactly what pushed a 16 GB card's
    # text-caching over the edge (surfaced by the VRAM simulator; real cards limped through
    # it on WDDM paging — the issue-#71 freeze). Same ops per element in the same order, so
    # still bit-identical — the transients are just bounded to a chunk. The scalar multiply
    # is in-place for the same reason.
    w_out = torch.empty(out, inp, dtype=torch.bfloat16, device=dev)
    chunk = max(1, (32 << 20) // max(inp, 1))                     # ~32M elements per chunk
    for r0 in range(0, out, chunk):
        r1 = min(out, r0 + chunk)
        codes = torch.empty(r1 - r0, inp, dtype=torch.uint8, device=dev)
        codes[:, 0::2] = packed[r0:r1] >> 4                       # HIGH nibble first (verified)
        codes[:, 1::2] = packed[r0:r1] & 0x0F
        vals = table[codes.long()]                                # one signed gather
        w = (vals.view(r1 - r0, inp // 16, 16) * bs[r0:r1].unsqueeze(-1)).mul_(gs)
        w_out[r0:r1] = w.view(r1 - r0, inp).to(torch.bfloat16)
    return w_out


def _dequant_comfy_weight(f, file_mod, ckpt):
    """Dequantize one comfy-quant Linear weight (nvfp4 or int8_tensorwise) back to bf16 [out, in].

    The `.comfy_quant` blob names the scheme — checked explicitly, because e.g. the
    int8_convrot TE variant stores ROTATED weights that would sail through a naive
    weight*scale dequant and silently produce a garbage encoder."""
    import json as _json
    fmt = ""
    try:
        blob = bytes(f.get_tensor(file_mod + ".comfy_quant").tolist())
        fmt = _json.loads(blob.decode("utf-8")).get("format", "")
    except Exception:
        pass
    if fmt not in ("nvfp4", "int8_tensorwise"):
        raise NotImplementedError(
            f"Unsupported comfy-quant format '{fmt or 'unknown'}' on {file_mod} — Fizgig can load "
            "the nvfp4-awq or bf16 Qwen3-VL-32B TE. The int8_convrot variant stores rotated "
            "weights and is not supported; use qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors "
            "(~15.7 GB) or the bf16 file instead.")
    if fmt == "nvfp4":
        packed = f.get_tensor(file_mod + ".weight")
        bscale = f.get_tensor(file_mod + ".weight_scale")
        gscale = f.get_tensor(file_mod + ".weight_scale_2")
        w = _nvfp4_dequant(packed, bscale, gscale)
        pqs_key = file_mod + ".pre_quant_scale"
        if pqs_key in ckpt:                               # AWQ: fold s into the weight columns
            pqs = f.get_tensor(pqs_key).to(torch.float32)
            w = (w.to(torch.float32) * pqs.unsqueeze(0)).to(torch.bfloat16)
        return w
    # int8_tensorwise: W = int8 * weight_scale (per-tensor scalar)
    w = f.get_tensor(file_mod + ".weight").to(torch.float32)
    s = f.get_tensor(file_mod + ".weight_scale").to(torch.float32)
    return (w * s).to(torch.bfloat16)


def _bundled_tokenizer_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "qwen3vl_tokenizer")


# H3 extends the stock Qwen3-VL vocabulary with these, in this order — MiniMax's own
# text_encoder/tokenizer_config.json lists them appended to Qwen's 13, which is what assigns them
# ids 151669..151675 (all inside the 151936-row embedding table, so they index real trained rows).
# ORDER IS LOAD-BEARING: they exist in neither vocab.json nor tokenizer.json, so transformers
# numbers them sequentially on load, and a reordering would silently shift every id.
_H3_SPECIAL_TOKENS = ("<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>",
                      "<|caption_start|>", "<|caption_end|>")


def _add_h3_special_tokens(tok):
    """Teach a stock Qwen3-VL tokenizer H3's markup, unless it already knows it.

    The bundled tokenizer is redistributed unmodified from Qwen3-VL-4B-Instruct and is SHARED
    with the krea2 encoder, so the tokens are added here rather than to that asset: patching the
    asset would change tokenization on an unrelated model path.

    Without this, `<d>` splits into byte pairs ([90707, 30768, ...] rather than [151669]) and
    `<|lyrics_start|>hold on<|lyrics_end|>` goes from 4 tokens to 16. Ordinary prose is
    unaffected, which is why the omission is invisible until a prompt carries dialogue markup —
    and H3 wraps ALL dialogue in `<d>[Language] ...</d>`.
    """
    missing = [t for t in _H3_SPECIAL_TOKENS if tok.convert_tokens_to_ids(t) is None]
    if missing:
        tok.add_tokens(missing, special_tokens=True)


def build_qwen3_te(config_overrides=None):
    """A Qwen3Model text encoder with no final norm (returns the raw layer-50 output).

    Text-only. For a caption with no reference image this is equivalent to the full Qwen3-VL
    stack: VL applies mrope, but every text token gets the SAME index on all three axes, which
    collapses mrope to ordinary 1-D rope. Reference images break that equality, which is why the
    r2v path needs build_qwen3vl_te() below rather than this."""
    from transformers import Qwen3Config, Qwen3Model
    cfg = dict(_QWEN3_32B_TRUNC50)
    if config_overrides:
        cfg.update(config_overrides)
    model = Qwen3Model(Qwen3Config(**cfg))
    model.norm = nn.Identity()          # comfy applies NO final norm to the layer-50 conditioning
    return model


# Qwen3-VL-32B vision tower. Every value here is transformers' own default for
# Qwen3VLVisionConfig EXCEPT out_hidden_size, and all of them were cross-checked against the
# shipped checkpoint's tensor shapes rather than taken on trust:
#   depth 27              <- 27 distinct visual.blocks.N
#   hidden_size 1152      <- visual.blocks.N.attn.proj.weight [1152, 1152]
#   intermediate_size 4304<- visual.blocks.N.mlp.linear_fc1.weight [4304, 1152]
#   patch_size 16,        <- visual.patch_embed.proj.weight [1152, 3, 2, 16, 16]
#   temporal_patch_size 2     (also gives in_channels 3 and the temporal patch)
#   spatial_merge_size 2  <- visual.merger.linear_fc1.weight in-features 4608 = 1152 * 2 * 2
#   out_hidden_size 5120  <- visual.merger.linear_fc2.weight [5120, 4608]; the DEFAULT IS 3584,
#                            which is the one value that would have been wrong left alone
#   num_position_embeddings 2304 <- visual.pos_embed.weight [2304, 1152]
#   deepstack_visual_indexes [8, 16, 24] <- 3 visual.deepstack_merger_list.N entries
# The config is hardcoded rather than fetched: ai-toolkit pulls it from the MiniMax HF repo,
# which is gated and geo-restricted, and Fizgig already hardcodes the text half the same way.
_QWEN3VL_VISION = dict(out_hidden_size=5120)

# Qwen3-VL is an mrope model, so the text stack needs a rope_scaling section. mrope_section sums
# to head_dim/2 (24+20+20 = 64 = 128/2).
_QWEN3VL_ROPE_SCALING = {"rope_type": "default", "mrope_section": [24, 20, 20],
                         "mrope_interleaved": True}


def build_qwen3vl_te(config_overrides=None):
    """The FULL Qwen3-VL stack — vision tower + language model — with no final norm.

    Needed only for reference (r2v) conditioning, where the prompt carries `<Picture i>:` vision
    blocks. Checked against the shipped nvfp4 checkpoint: all 902 of its base tensors map onto
    this module tree with nothing left over (see tests/test_minimax_ref_encoder.py). The three
    parameters with no checkpoint entry are `language_model.norm.weight` (replaced by Identity
    here, as comfy applies no final norm) and the two computed rope inv_freq buffers."""
    from transformers import Qwen3VLConfig, Qwen3VLModel
    txt = dict(_QWEN3_32B_TRUNC50)
    txt["rope_scaling"] = dict(_QWEN3VL_ROPE_SCALING)
    vis = dict(_QWEN3VL_VISION)
    if config_overrides:
        txt.update(config_overrides.get("text", {}))
        vis.update(config_overrides.get("vision", {}))
    model = Qwen3VLModel(Qwen3VLConfig(text_config=txt, vision_config=vis))
    model.language_model.norm = nn.Identity()
    return model


# AdaLN modality tags. The DiT selects its modulation by these, so a mis-tagged row is modulated
# as the wrong modality — it renders, it just renders wrong.
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2


def build_image_processor(tokenizer_dir=None):
    """The Qwen3-VL image processor, from the config bundled in src/fizgig/assets.

    Built directly from the bundled preprocessor_config.json rather than downloaded: MiniMax's
    HF repo is gated and geo-restricted, and everything needed (patch 16, temporal patch 2,
    merge 2, mean/std 0.5) already ships with the tokenizer assets."""
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(tokenizer_dir or _bundled_tokenizer_dir())


def build_reference_tokens(tokenizer, image_processor, caption, images, max_length: int = None):
    """The r2v prompt presentation: `<Picture i>: <vision block>` per image, then the caption.

    Returns (input_ids [1, L], token_tags [L], pixel_values, image_grid_thw). Mirrors
    ai-toolkit's encode_minimax_h3_prompt and comfy's MiniMaxH3Tokenizer.tokenize_with_weights:
    raw tokens, no chat template, no special tokens.

    The tags are the load-bearing part. `<Picture 1>: ` is TEXT, but the WHOLE vision run —
    the <|vision_start|> and <|vision_end|> markers included, not just the image pads — is
    VIDEO. Tagging the markers as text would modulate two rows per image as the wrong modality.
    """
    pixel_values, image_grid_thw = None, None
    ids, tags = [], []
    if images:
        vision = image_processor(images=images, return_tensors="pt")
        pixel_values, image_grid_thw = vision["pixel_values"], vision["image_grid_thw"]
        merge = image_processor.merge_size ** 2
        v_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        v_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        img_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        for i in range(len(images)):
            n_img = int(image_grid_thw[i].prod()) // merge
            label = tokenizer(f"<Picture {i + 1}>: ", add_special_tokens=False)["input_ids"]
            vision_ids = [v_start] + [img_pad] * n_img + [v_end]
            ids += label + vision_ids
            tags += [TEXT_TAG] * len(label) + [VIDEO_TAG] * len(vision_ids)

    _tk = dict(add_special_tokens=False)
    if max_length:                                  # None -> no cap, matching ComfyUI
        _tk.update(truncation=True, max_length=max_length)
    prompt_ids = tokenizer(caption, **_tk)["input_ids"]
    ids += prompt_ids
    tags += [TEXT_TAG] * len(prompt_ids)
    if not ids:                                     # empty prompt with no reference
        pid = getattr(tokenizer, "pad_token_id", None)
        ids, tags = [151643 if pid is None else int(pid)], [TEXT_TAG]
    return (torch.tensor([ids], dtype=torch.long), torch.tensor(tags, dtype=torch.long),
            pixel_values, image_grid_thw)


def qwen3vl_key_map(key: str) -> str:
    """Checkpoint key -> Qwen3VLModel parameter name.

    The single-file checkpoint stores the language stack under `model.` and the vision tower
    under `visual.`; Qwen3VLModel wants `language_model.` and `visual.`."""
    return "language_model." + key[len("model."):] if key.startswith("model.") else key


class MiniMaxH3TextEncoder:
    """Loads the bf16 TE, NF4 on GPU, and encodes captions to [1, L, 5120] bf16."""

    def __init__(self, model, tokenizer, device="cuda", compute_dtype=torch.bfloat16,
                 cpu_embed=False):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.compute_dtype = compute_dtype
        self.cpu_embed = cpu_embed     # embed_tokens left on CPU; see _text_forward()
        self._cache = {}          # caption -> CPU embedding; see encode()
        self._image_processor = None   # built on first r2v encode; see encode_with_reference()

    def _text_forward(self, ids):
        """Token ids -> last_hidden_state, on whichever side the embedding table lives.

        cpu_embed=False is the original call verbatim: ids to the device, `input_ids=`.

        cpu_embed=True leaves the 151936 x 5120 table on the CPU — ~1.5 GB of VRAM that served
        one lookup per forward — so the gather happens there and only its `[B, L, 5120]` result
        crosses to the GPU, passed as `inputs_embeds=`. The decoder stack sees the identical
        tensor either way; a gather is memory-bound and trivial at caption lengths.

        Text-only path. The vision build scatters image embeddings into the `<|image_pad|>` slots
        from input_ids, so it never takes the cpu_embed branch (see load_minimax_h3_te)."""
        if not self.cpu_embed:
            return self.model(input_ids=ids.to(self.device)).last_hidden_state
        emb = self.model.embed_tokens(ids.to("cpu")).to(self.device)
        return self.model(inputs_embeds=emb).last_hidden_state

    def _pad_id(self) -> int:
        """The tokenizer's pad id, or Qwen's default. NOT `pad_token_id or <default>` — a pad id
        of 0 is falsy and would silently be replaced by the default, which is out of range for
        any tokenizer with a smaller vocab."""
        pid = getattr(self.tokenizer, "pad_token_id", None)
        return 151643 if pid is None else int(pid)

    @torch.no_grad()
    def encode(self, caption: str, max_length: int = None) -> torch.Tensor:
        """Encode one caption to [1, L, 5120]. Memoized by caption text for the caching pass.

        Text conditioning does not depend on resolution or on anything else that varies between
        dataset blocks, so a caption that appears more than once — repeated text, or several
        dataset entries over the same folder — is encoded once. Under the nvfp4-resident encoder
        a forward dequantizes 351 weights, so a repeat is far from free.

        max_length=None means NO TRUNCATION, matching ComfyUI (its MiniMax tokenizer sets
        max_length=99999999, pad_to_max_length=False). This used to cap at 512, which silently
        dropped the tail of a long prompt — and worse, the text length sets the media clock
        ORIGIN for the video rows, so a truncated prompt also shifted the render onto a
        different temporal grid than the same prompt in ComfyUI."""
        hit = self._cache.get(caption)
        if hit is not None:
            return hit.clone()                             # callers must not share storage
        # H3: raw prompt text, NO special tokens (no chat template).
        _tk = dict(add_special_tokens=False, return_tensors="pt")
        if max_length:                                     # None/0 -> no cap, as ComfyUI does
            _tk.update(truncation=True, max_length=max_length)
        ids = self.tokenizer(caption, **_tk)["input_ids"]
        if ids.shape[1] == 0:                              # empty caption -> single pad token
            ids = torch.tensor([[self._pad_id()]])
        # norm=Identity -> raw layer-50 output
        emb = self._text_forward(ids).to(self.compute_dtype)  # [1, L, 5120]
        # keep it on CPU: a few hundred KB per caption, and GPU memory is the scarce thing here
        self._cache[caption] = emb.detach().to("cpu")
        return emb

    @torch.no_grad()
    def encode_with_reference(self, caption: str, images, max_length: int = None):
        """r2v conditioning: `<Picture i>:` vision blocks + caption -> ([1, L, 5120], tags [L]).

        Requires the encoder to have been built with build_qwen3vl_te() — the text-only
        Qwen3Model has no vision tower and would silently ignore the pixels. NOT memoized: the
        cache is keyed by caption text alone, which would collide across different reference
        images for the same caption.
        """
        if not hasattr(self.model, "visual"):
            raise RuntimeError(
                "encode_with_reference needs the vision-capable encoder — load with "
                "with_vision=True (build_qwen3vl_te), not the text-only Qwen3Model.")
        if self._image_processor is None:
            self._image_processor = build_image_processor()
        ids, tags, pixel_values, grid = build_reference_tokens(
            self.tokenizer, self._image_processor, caption, images, max_length)
        kw = {}
        if pixel_values is not None:
            # the vision tower is bf16 in the checkpoint; feed it its own dtype
            kw["pixel_values"] = pixel_values.to(self.device, torch.bfloat16)
            kw["image_grid_thw"] = grid.to(self.device)
        out = self.model(input_ids=ids.to(self.device),
                         attention_mask=torch.ones_like(ids).to(self.device), **kw)
        # norm is Identity on this build, so last_hidden_state IS the raw layer-50 conditioning
        emb = out.last_hidden_state.to(self.compute_dtype)
        return emb, tags

    # NO encode_with_reference_batch. Batching the reference encodes was implemented and
    # MEASURED against the one-at-a-time path on the real encoder (tests/diag_ref_batch_encode.py)
    # and it is NOT equivalent: max|diff| 1.3e2 to 1.7e3 on conditioning whose values top out
    # around 2e4, i.e. up to ~8% — against a left-padding control of 1.8e4. A proper 0/1 attention
    # mask (Qwen3-VL derives its mrope positions from it) improved matters but did not fix them.
    #
    # Right padding IS exact on the caption path, for the causal-attention reason documented on
    # encode_batch. The vision path adds mrope positions computed from image_grid_thw and a
    # scatter into the <|image_pad|> slots, and something there does not survive padding. Rather
    # than ship conditioning that is subtly wrong in a way nothing downstream would reveal, the
    # reference pass stays one encode at a time. The cost is ~1.5 s per encode — about 2.5 min
    # for 46 images at 2 references, linear in (images x references), and cached afterwards.
    #
    # If this is ever worth revisiting, the diagnostic is the gate: it must reach the ~1e-7 the
    # caption path achieves, not merely "look close".

    @torch.no_grad()
    def encode_batch(self, captions, max_length: int = None, batch_size: int = 8):
        """Encode many captions, returning [1, L_i, 5120] each — same values as encode(), fewer
        forwards.

        Under the nvfp4-resident encoder a forward dequantizes 351 weights, and that cost is per
        FORWARD, not per token: batching amortizes it across the batch. The rest is exactness.

        RIGHT padding, no attention mask needed. The stack is causal, so a real token at position
        i attends only to 0..i and trailing pads cannot influence it; slicing each row back to its
        true length recovers the single-caption result. Verified on a real Qwen3 stack
        (tests/diag_batch_encode.py): max|diff| 4.5e-08 across batch sizes and pad ids, while the
        LEFT-padding control diverges by 1.9e-01 — the equivalence is a property of right padding
        specifically, not of batching in general."""
        out = [None] * len(captions)
        todo = [i for i, c in enumerate(captions) if c not in self._cache]
        for i, c in enumerate(captions):
            if c in self._cache:
                out[i] = self._cache[c].clone()

        pad_id = self._pad_id()
        for start in range(0, len(todo), batch_size):
            idxs = todo[start:start + batch_size]
            toks = []
            for i in idxs:
                _tk = dict(add_special_tokens=False, return_tensors="pt")
                if max_length:                      # None -> no cap, matching ComfyUI
                    _tk.update(truncation=True, max_length=max_length)
                t = self.tokenizer(captions[i], **_tk)["input_ids"][0]
                toks.append(t if t.numel() else torch.tensor([pad_id]))
            L = max(t.numel() for t in toks)
            ids = torch.full((len(toks), L), pad_id, dtype=torch.long)
            for r, t in enumerate(toks):
                ids[r, : t.numel()] = t
            hs = self._text_forward(ids)
            for r, i in enumerate(idxs):
                emb = hs[r, : toks[r].numel()].unsqueeze(0).to(self.compute_dtype)
                self._cache[captions[i]] = emb.detach().to("cpu")
                out[i] = emb
        return out


class Nvfp4Linear(nn.Linear):
    """A frozen Linear that KEEPS the checkpoint's nvfp4 weights, like the reference loader.

    The previous path dequantized nvfp4 -> bf16 and then re-quantized to NF4, which adds ~9%
    relative error to the encoder (measured per tensor on the real file) that the reference
    never incurs — it attaches the packed nvfp4 tensors and uses them directly. NF4 is also the
    coarser format here: one absmax per 64 elements against nvfp4's FP8 scale per 16, plus AWQ.

    Cost of keeping them: nothing. Packed nvfp4 is 0.5 byte/param, the same as NF4 — ~15.7 GB
    for this encoder either way. Dequantization moves to the matmul, which is fine because the
    TE runs once per caption during the caching pass, not per training step.

    AWQ note: `pre_quant_scale` multiplies the INPUT (comfy ops.py does `input * s`), so it is
    applied to the activation here rather than folded into the weight columns — exact, and it
    keeps the stored block scales untouched.
        Both scale tensors are held as their raw BYTES (uint8 views) and reinterpreted in the
    forward. The block scales are float8_e4m3fn and the global scale fp32; a stray
    `.to(dtype=...)` anywhere would silently cast either one — for the fp8 scales that is not
    lossy, it is garbage, since a cast reads them as numbers rather than reinterpreting. The
    byte view cannot be cast.
    """

    def __init__(self, in_features, out_features, bias=False, compute_dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias=bias)
        del self._parameters["weight"]
        self.compute_dtype = compute_dtype
        self.register_buffer("packed", torch.empty(out_features, in_features // 2,
                                                   dtype=torch.uint8), persistent=False)
        # fp8 block scales [out, in/16], stored as bytes
        self.register_buffer("bscale", torch.empty(out_features, in_features // 16,
                                                   dtype=torch.uint8), persistent=False)
        # fp32 scalar, stored as its 4 bytes
        self.register_buffer("gscale", torch.empty(4, dtype=torch.uint8), persistent=False)
        self.register_buffer("pre_quant_scale", None, persistent=False)

    def _scales(self):
        return self.bscale.view(torch.float8_e4m3fn), self.gscale.view(torch.float32)

    @property
    def weight(self):
        """The TRUE dense weight, materialized on demand (inspection only — the forward
        dequantizes inline). The AWQ scale is folded into the columns here so this matches the
        weight the reference produces; the forward applies it to the activation instead, which
        is algebraically identical and one rounding step shorter."""
        bs, gs = self._scales()
        w = _nvfp4_dequant(self.packed, bs, gs).to(self.compute_dtype)
        if self.pre_quant_scale is not None:
            w = w * self.pre_quant_scale.to(w.dtype).reshape(1, -1)
        return w

    def forward(self, x):
        dt = self.compute_dtype
        if self.pre_quant_scale is not None:
            x = x * self.pre_quant_scale.to(x.dtype)      # AWQ acts on the input
        bs, gs = self._scales()
        return torch.nn.functional.linear(x.to(dt), _nvfp4_dequant(self.packed, bs, gs).to(dt),
                                          self.bias)


def load_minimax_h3_te(path: str, device="cuda", compute_dtype=torch.bfloat16,
                       quantize=True, tokenizer_dir=None,
                       te_quant="auto", with_vision=False,
                       cpu_embed=True) -> MiniMaxH3TextEncoder:
    """Build the Qwen3-VL-32B TE. Language-only by default (visual.* skipped).

    cpu_embed keeps the ~1.5 GB embedding table in system RAM and gathers there, which is free
    at caption lengths and is the difference between fitting a 16 GB card and being paged out by
    the driver. It is forced OFF with the vision tower: that build scatters image embeddings into
    the `<|image_pad|>` slots from input_ids, and feeding it inputs_embeds would drop the pixels
    silently.

    with_vision=True builds the FULL stack instead — needed for r2v reference conditioning,
    where the prompt carries `<Picture i>` vision blocks. The vision tower is left in bf16: it
    is ~176 weights against the language stack's thousands, and none of its module names match
    the quantization suffixes (checked — `attn.qkv` / `mlp.linear_fc1`, not `self_attn.q_proj`
    / `mlp.gate_proj`), so it is excluded automatically rather than by a special case.

    te_quant:
      "nvfp4" — KEEP the checkpoint's packed nvfp4 weights (what the reference does). Same
                residency as NF4 (~0.5 byte/param) with none of the extra ~9% error.
      "nf4"   — dequantize to bf16 then 4-bit quantize with bitsandbytes. Required for the
                bf16 checkpoint, which has nothing to keep.
      "auto"  — nvfp4 when the file is nvfp4-awq, else nf4.
    """
    from bitsandbytes.nn import Linear4bit, Params4bit
    from transformers import AutoTokenizer

    cpu_embed = bool(cpu_embed) and not with_vision

    with MemoryEfficientSafeOpen(path) as _probe:
        _is_cq = any(k.endswith(".comfy_quant") for k in _probe.keys())
    mode = te_quant
    if mode == "auto":
        mode = "nvfp4" if _is_cq else "nf4"
    if mode == "nvfp4" and not _is_cq:
        raise ValueError("te_quant='nvfp4' needs the nvfp4-awq checkpoint")
    if not quantize:
        mode = "none"

    # Parameter name -> checkpoint key. The file stores the language stack under `model.` and
    # the vision tower under `visual.`; the VL module tree calls those `language_model.` and
    # `visual.`. Getting this wrong does not raise — the parameter simply keeps its random init.
    def _ck(name: str) -> str:
        if not with_vision:
            return "model." + name
        if name.startswith("language_model."):
            return "model." + name[len("language_model."):]
        return name

    with torch.device("meta"):
        model = build_qwen3vl_te() if with_vision else build_qwen3_te()

        # Swap the NF4-target Linears for Linear4bit shells INSIDE the meta context. Outside it,
        # each Linear4bit eagerly allocates a full fp32 CPU weight (nn.Linear default) — across the
        # 32B Qwen3-VL that is well over 100 GB of throwaway tensors the allocator then holds for
        # the whole caching pass. On meta the shells are 0 bytes; real weights stream in below.
        if mode != "none":
            for mod_name, module in list(model.named_modules()):
                for child_name, child in list(module.named_children()):
                    full = f"{mod_name}.{child_name}" if mod_name else child_name
                    if not (isinstance(child, nn.Linear)
                            and (full + ".weight").endswith(_NF4_SUFFIXES)):
                        continue
                    if mode == "nvfp4":
                        setattr(module, child_name,
                                Nvfp4Linear(child.in_features, child.out_features,
                                            bias=child.bias is not None,
                                            compute_dtype=compute_dtype))
                    else:
                        q = Linear4bit(child.in_features, child.out_features,
                                       bias=child.bias is not None,
                                       compute_dtype=compute_dtype, quant_type="nf4")
                        setattr(module, child_name, q)

    dev = torch.device(device)
    print("[load] streaming the Qwen3-VL-32B text encoder — a couple of quiet minutes here is "
          "normal (nvfp4 dequant is the slower one; a bitsandbytes 'expandable_segments not "
          "supported' warning on Windows is harmless).", flush=True)
    model_keys = {n for n, _ in model.named_parameters()}
    with MemoryEfficientSafeOpen(path) as f:
        ckpt = set(f.keys())
        # The comfy nvfp4-awq TE (15 GB) stores packed quant weights + scales instead of plain
        # bf16 — detect once, then dequantize each Linear weight from its
        # `<mod>.weight/.weight_scale[/_2]/.pre_quant_scale` family back to bf16. Everything
        # else (norms, embed) loads as usual; the checkpoint's norm weights already carry the
        # folded AWQ scales, so the model function matches the bf16 TE up to 4-bit noise.
        is_comfy_quant = any(k.endswith(".comfy_quant") for k in ckpt)
        if mode == "nvfp4":
            print("[minimax-te] comfy-quant checkpoint (nvfp4-awq) — KEEPING the packed nvfp4 "
                  "weights (the reference's own storage; NF4 on top would add ~9% error)")
        elif is_comfy_quant:
            print("[minimax-te] comfy-quant checkpoint (nvfp4-awq) — dequantizing to bf16 per tensor")
        # nvfp4 mode: the quantized linears hold BUFFERS, not parameters — fill them first.
        if mode == "nvfp4":
            for mod_name, module in model.named_modules():
                if not isinstance(module, Nvfp4Linear):
                    continue
                fm = _ck(mod_name)
                module.packed = f.get_tensor(fm + ".weight").to(torch.uint8).to(dev)
                # byte views: fp8 block scales and the fp32 global scale must never be CAST
                module.bscale = f.get_tensor(fm + ".weight_scale").contiguous().view(
                    torch.uint8).to(dev)
                module.gscale = f.get_tensor(fm + ".weight_scale_2").to(torch.float32).reshape(
                    1).contiguous().view(torch.uint8).to(dev)
                pqs = fm + ".pre_quant_scale"
                module.pre_quant_scale = (f.get_tensor(pqs).to(compute_dtype).to(dev)
                                          if pqs in ckpt else None)

        for name in model_keys:
            src = _ck(name)
            file_mod = src.rsplit(".", 1)[0]
            leaf = name.rsplit(".", 1)[1]
            if is_comfy_quant and leaf == "weight" and (file_mod + ".comfy_quant") in ckpt:
                w = _dequant_comfy_weight(f, file_mod, ckpt)   # -> bf16 [out, in]
            elif src in ckpt:
                w = f.get_tensor(src)
            else:
                continue                                   # e.g. norm (Identity) has no params
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            if mode == "nvfp4" and name.endswith(_NF4_SUFFIXES):
                continue                                   # placed as nvfp4 buffers above
            if mode == "nf4" and (name.endswith(_NF4_SUFFIXES)):
                p = Params4bit(w.to(compute_dtype), requires_grad=False, quant_type="nf4").to(dev)
                setattr(parent, leaf, p)
            else:
                keep = w.to(torch.float32) if w.dtype == torch.float32 else w.to(compute_dtype)
                # embed_tokens is excluded from _NF4_SUFFIXES on purpose, so it lands unquantized
                # — 151936 x 5120 in bf16 is ~1.5 GB. It serves one gather per forward, so on the
                # text-only path it stays in system RAM and never reaches the card at all. That
                # margin is what lets a 16 GB card clear the caching pass without the driver
                # paging the encoder back out over PCIe.
                tgt = torch.device("cpu") if (cpu_embed and name == "embed_tokens.weight") else dev
                setattr(parent, leaf, nn.Parameter(keep.to(tgt), requires_grad=False))

    # Computed (non-checkpoint) buffers stayed on meta from the meta-build. The rotary
    # embedding's inv_freq is the load-bearing one — rebuild it on the real device. A general
    # sweep materializes any other stray meta buffers to be safe.
    # These carry inv_freq, which the generic sweep below would otherwise ZERO — a rope table of
    # zeros is no positional information at all, and the model would still run.
    if with_vision:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (Qwen3VLTextRotaryEmbedding,
                                                                    Qwen3VLVisionRotaryEmbedding)
        model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(model.config.text_config).to(dev)
        _vcfg = model.config.vision_config
        model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(
            _vcfg.hidden_size // _vcfg.num_heads // 2).to(dev)
    else:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        model.rotary_emb = Qwen3RotaryEmbedding(model.config).to(dev)
    for mod in model.modules():
        for bname, buf in list(mod.named_buffers(recurse=False)):
            if buf is not None and buf.is_meta:
                mod.register_buffer(bname, torch.zeros(buf.shape, dtype=buf.dtype, device=dev))
    model.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained(tokenizer_dir or _bundled_tokenizer_dir())
    _add_h3_special_tokens(tok)
    return MiniMaxH3TextEncoder(model, tok, device=device, compute_dtype=compute_dtype,
                                cpu_embed=cpu_embed)
