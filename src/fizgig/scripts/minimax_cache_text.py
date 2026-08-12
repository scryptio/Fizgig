"""Cache Qwen3-VL-32B text-encoder states (raw layer-50 output) for MiniMax H3 image training.

    python src/fizgig/scripts/minimax_cache_text.py --dataset_config config.toml --text_encoder path/to/qwen3vl_32b_bf16
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.dataset.config import (
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
    load_user_config,
)
from fizgig.scripts.cache_text import prepare_cache_files_and_paths, process_batches, post_process
from fizgig.minimax.embedder import load_minimax_h3_te
from fizgig.minimax.caching import encode_and_save_text
from fizgig.training.metadata import ARCHITECTURE_MINIMAX

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache Qwen3-VL-32B text-encoder states for MiniMax H3 training")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config .toml file")
    parser.add_argument("--text_encoder", type=str, required=True, help="Path to the bf16 Qwen3-VL-32B safetensors")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Captions per text-encoder forward. The nvfp4-resident encoder "
                             "dequantizes 351 weights per FORWARD, so this is the dial that "
                             "matters: 1 caption/forward costs ~1.2 s, 16 costs barely more. "
                             "Right-padding makes batching exactly equivalent to one-at-a-time "
                             "for this causal stack (tests/diag_batch_encode.py).")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="Keep stale cache files")
    parser.add_argument("--no_quantize", action="store_true", help="Load the TE in bf16 (no NF4) — needs ~66 GB VRAM")
    parser.add_argument("--reference_count", type=int, default=0, metavar="K",
                        help="Reference distillation: also cache r2v conditioning for the "
                             "teacher, pairing each image with K OTHER images from this same "
                             "dataset. 0 disables. Every image ends up used as a reference K "
                             "times; no image is ever its own reference (the teacher would just "
                             "copy the answer). Needs the vision tower, so the TE loads ~1 GB "
                             "larger, and needs latents cached first.")
    return parser


def _cache_reference_conditioning(args, datasets, all_files, all_paths, encoder):
    """Pair every image with K OTHER images and cache the teacher's conditioning for each.

    Pairing is a fixed ROTATION — item i references items i+1..i+K (mod N). Deterministic, so a
    re-cache reproduces it and two runs stay comparable; self-excluding by construction, which is
    the property that matters most (an image shown itself lets the teacher copy the answer, and
    the distillation would collapse back to photo reconstruction).

    The reference LATENT is the other image's ALREADY-CACHED latent — item j's latent simply is
    the VAE encoding of image j — so nothing is encoded twice and no VAE is needed here. The
    vision block therefore has to describe that same picture, so the image is resized to the
    latent's implied pixel size (h*16 x w*16) rather than to a canvas of its own."""
    import glob as _glob

    from fizgig.dataset.image_dataset import MemoryEfficientSafeOpen
    from fizgig.minimax.caching import encode_and_save_reference_text
    from fizgig.minimax.reference import reference_image_for_latent

    k = max(0, int(args.reference_count or 0))
    for ds_i, ds in enumerate(datasets):
        paths = list(getattr(getattr(ds, "datasource", None), "image_paths", []) or [])
        n = len(paths)
        if n < 2:
            logger.warning("[reference] dataset %d has %d image(s) — need at least 2 to pair "
                           "without an image referencing itself; skipping", ds_i, n)
            continue
        kk = min(k, n - 1)
        if kk < k:
            logger.warning("[reference] only %d other image(s) available; using %d reference(s) "
                           "per image instead of %d", n - 1, kk, k)

        # item_key -> index, so an ItemInfo can find its own position in the rotation
        keys = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        idx_of = {kx: i for i, kx in enumerate(keys)}
        cache_dir = getattr(ds, "cache_directory", "") or ""

        def _latent_for(i):
            """The cached latent of image i."""
            hits = _glob.glob(os.path.join(_glob.escape(cache_dir),
                                           f"{_glob.escape(keys[i])}_*{ARCHITECTURE_MINIMAX}.safetensors"))
            if not hits:
                raise SystemExit(
                    f"[reference] no cached latent for {keys[i]} — run latent caching first. "
                    f"The reference latent is reused from that cache rather than encoded again.")
            with MemoryEfficientSafeOpen(hits[0]) as f:
                key = next(kx for kx in f.keys() if kx.startswith("latent_"))
                return f.get_tensor(key)

        _img_cache, _lat_cache = {}, {}

        def reference_for(item):
            i = idx_of.get(item.item_key)
            if i is None:                       # key shapes differ -> match on the basename
                i = idx_of.get(os.path.splitext(os.path.basename(str(item.item_key)))[0])
            if i is None:
                logger.warning("[reference] %s is not in the image list; skipped", item.item_key)
                return []
            out = []
            for slot in range(kk):
                j = (i + 1 + slot) % n          # never i itself
                if j not in _lat_cache:
                    z = _latent_for(j)
                    _lat_cache[j] = z
                    # framed exactly as the picture this latent was encoded from — same helper
                    # the trainer uses, so a crop can never become a stretch
                    _img_cache[j] = reference_image_for_latent(paths[j], z)
                out.append((slot, _img_cache[j], _lat_cache[j], os.path.basename(paths[j])))
            return out

        logger.info("[reference] dataset %d: %d images, %d reference(s) each (rotation) — every "
                    "image is used as a reference %d time(s)", ds_i, n, kk, kk)

        # Clear this dataset's OLD reference slots first. The trainer picks a slot by existence
        # (image_dataset: `[i for i in range(8) if os.path.exists(...ref{i}...)]`), so re-caching
        # with FEWER references used to leave the higher-numbered files behind and they could
        # still be drawn — silently training against a pairing from a previous configuration.
        # Safe because the pass below rewrites slots 0..kk-1 for every image in this dataset;
        # nothing is deleted that is not about to be regenerated. Scoped to this cache dir, so a
        # second subject's references are untouched.
        if cache_dir and os.path.isdir(cache_dir):
            _stale = [p for p in _glob.glob(os.path.join(_glob.escape(cache_dir),
                                                         f"*{ARCHITECTURE_MINIMAX}_teref*.safetensors"))]
            for _p in _stale:
                try:
                    os.remove(_p)
                except OSError as _e:
                    logger.warning("[reference] could not clear %s (%s)", _p, _e)
            if _stale:
                logger.info("[reference] cleared %d stale reference slot file(s) in %s",
                            len(_stale), cache_dir)

        process_batches(args, [ds], [all_files[ds_i]], [all_paths[ds_i]],
                        lambda batch: encode_and_save_reference_text(encoder, batch, reference_for),
                        index_offset=ds_i)


def main():
    args = setup_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint_gen = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Loading dataset config from {args.dataset_config}")
    user_config = load_user_config(args.dataset_config)
    blueprint = blueprint_gen.generate(user_config, args, architecture=ARCHITECTURE_MINIMAX)
    datasets = generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets

    all_files, all_paths = prepare_cache_files_and_paths(datasets)

    _refs = max(0, int(args.reference_count or 0))
    logger.info(f"Loading Qwen3-VL-32B text encoder from {args.text_encoder}"
                + (f" (with the vision tower — {_refs} reference(s) per image)" if _refs else ""))
    encoder = load_minimax_h3_te(args.text_encoder, device=device, compute_dtype=torch.bfloat16,
                                 quantize=not args.no_quantize, with_vision=bool(_refs))

    process_batches(args, datasets, all_files, all_paths, lambda batch: encode_and_save_text(encoder, batch))

    # r2v conditioning for the teacher. Written AFTER the plain cache and to sibling files, so
    # a run without --reference_count is byte-identical to before.
    if _refs:
        _cache_reference_conditioning(args, datasets, all_files, all_paths, encoder)

    _uncond = encoder.encode("")[0].detach().cpu().contiguous()      # (L=1, 5120)
    del encoder
    post_process(datasets, all_files, all_paths, args.keep_cache)

    # One UNCONDITIONAL embed per cache dir (empty prompt -> the TE's single-pad fallback), so
    # caption dropout has something to swap in for ~5% of steps.
    #
    # Written AFTER post_process, deliberately. Its filename ends `_minimaxh3_te.safetensors`, so
    # it matches the glob post_process uses to find stale caches — and since no dataset ITEM
    # claims that path, a second caching pass would delete the one the first pass wrote. The GUI
    # re-runs caching on every launch, so the file existed only until the next run and
    # ss_caption_dropout silently read 0.
    from safetensors.torch import save_file as _save_file
    _n = 0
    for ds in datasets:
        _dir = getattr(ds, "cache_directory", None)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
            _save_file({"hidden_states": _uncond,
                        "attention_mask": torch.ones(_uncond.shape[0], dtype=torch.bool)},
                       os.path.join(_dir, f"uncond_{ARCHITECTURE_MINIMAX}_te.safetensors"))
            _n += 1
    logger.info(f"[uncond] cached the empty-prompt embed for caption dropout "
                f"({tuple(_uncond.shape)}) in {_n} cache dir(s)")


if __name__ == "__main__":
    main()
