"""LoRA SFT on the training traces, with the Fireworks cookbook's SFT loop (`training/recipes/sft_loop.py`).

The loop lives in the cookbook, which is not part of this package. Set it up as in rl_train.py:

    git clone https://github.com/fw-ai/cookbook.git
    cd cookbook/training && pip install -e .        # needs Python >= 3.11
    export PYTHONPATH=/path/to/cookbook:$PYTHONPATH  # the cookbook root holding `training/`

Assistant tokens are the only loss target (thinking and label). Each training row has one
assistant turn, so `all_assistant_messages` is the assistant-only loss. Rows are rendered with the
cookbook's Qwen3 renderer (thinking on), the same template the eval uses for Qwen3 thinking.

A checkpoint is saved after every epoch: `sampler_save_interval` is set to the number of optimizer
steps in one epoch (ceil(rows / batch size)). The cookbook's step counter runs across epochs, so
step k * interval is the end of epoch k.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

DEFAULT_DATASET = Path("outputs/qwen3-4b/zero-shot-v3/sft_data/train_sft.jsonl")
DEFAULT_LOG_PATH = Path("outputs/qwen3-4b/zero-shot-v3/sft_logs")
BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
TOKENIZER_MODEL = "Qwen/Qwen3-4B"  # the same tokenizer the eval and RL use
RENDERER_NAME = "qwen3"  # cookbook renderer for Qwen3 with thinking on
TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3-4b-minimum"  # the RL run's shape
TRAIN_ON_WHAT = "all_assistant_messages"


def count_rows(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def steps_per_epoch(rows: int, batch_size: int) -> int:
    return math.ceil(rows / batch_size)


def build_config(args: argparse.Namespace) -> Any:
    from recipes.sft_loop import Config  # cookbook import, needs PYTHONPATH
    from training.utils import TrainerConfig, WandBConfig

    rows = count_rows(args.dataset)
    return Config(
        log_path=str(args.log_path),
        base_model=BASE_MODEL,
        dataset=str(args.dataset),
        tokenizer_model=TOKENIZER_MODEL,
        renderer_name=RENDERER_NAME,
        train_on_what=TRAIN_ON_WHAT,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        lora_rank=args.lora_rank,
        sampler_save_interval=steps_per_epoch(rows, args.batch_size),
        save_final_checkpoint=False,  # the last epoch's sampler checkpoint is the final one
        output_model_id=args.output_model_id,
        trainer=TrainerConfig(training_shape_id=args.training_shape),
        wandb=WandBConfig(project="anli-sft"),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT on the training traces (cookbook sft_loop).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64, help="sequences per optimizer step")
    parser.add_argument("--max-seq-len", type=int, default=4600)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--training-shape", default=TRAINING_SHAPE)
    parser.add_argument("--output-model-id", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the config and exit without training")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows = count_rows(args.dataset)
    print(json.dumps({
        "dataset": str(args.dataset),
        "rows": rows,
        "steps_per_epoch": steps_per_epoch(rows, args.batch_size),
        "total_steps": steps_per_epoch(rows, args.batch_size) * args.epochs,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "lora_rank": args.lora_rank,
        "renderer_name": RENDERER_NAME,
        "train_on_what": TRAIN_ON_WHAT,
    }, indent=2))
    if args.dry_run:
        return
    from recipes.sft_loop import main as run_sft

    run_sft(build_config(args))


if __name__ == "__main__":
    main()
