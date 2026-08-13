"""Cache Qwen3-VL-4B text-encoder outputs (multi-layer hidden stack + mask) for Krea 2 training.

    python src/fizgig/scripts/krea2_cache_text.py --dataset_config config.toml --text_encoder path/to/qwen3vl_4b_bf16
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
from fizgig.krea2.utils import load_krea2_text_encoder
from fizgig.krea2.caching import encode_and_save_text
from fizgig.training.metadata import ARCHITECTURE_KREA2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache Qwen3-VL-4B text-encoder outputs for Krea 2 training")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config .toml file")
    parser.add_argument("--text_encoder", type=str, required=True, help="Path to the bf16 Qwen3-VL-4B safetensors")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for encoding")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="Keep stale cache files")
    return parser


def main():
    args = setup_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint_gen = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Loading dataset config from {args.dataset_config}")
    user_config = load_user_config(args.dataset_config)
    blueprint = blueprint_gen.generate(user_config, args, architecture=ARCHITECTURE_KREA2)
    datasets = generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets

    all_files, all_paths = prepare_cache_files_and_paths(datasets)

    logger.info(f"Loading Qwen3-VL-4B text encoder from {args.text_encoder}")
    encoder = load_krea2_text_encoder(args.text_encoder, dtype=torch.bfloat16, device=device)

    process_batches(args, datasets, all_files, all_paths, lambda batch: encode_and_save_text(encoder, batch))
    del encoder
    post_process(datasets, all_files, all_paths, args.keep_cache)


if __name__ == "__main__":
    if sys.platform == "linux" and os.environ.get("FIZGIG_GPU_BACKEND", "").lower() == "rocm":
        from fizgig.rocm.cache_exit import run_cache_main

        run_cache_main(main)
    else:
        main()
