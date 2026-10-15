"""LoRA SFT on the training traces, with the Fireworks cookbook's SFT loop (`training/recipes/sft_loop.py`).

The loop lives in the cookbook, which is not part of this package. Set it up as in rl_train.py:

    git clone https://github.com/fw-ai/cookbook.git
    cd cookbook/training && pip install -e .        # needs Python >= 3.11
    export PYTHONPATH=/path/to/cookbook:/path/to/cookbook/training:$PYTHONPATH

Settings:
- Loss on all assistant tokens (thinking and label). Each row has one assistant turn.
- Rows are rendered with the cookbook's Qwen3 renderer (thinking on), the eval's Qwen3 template.
- eval_auto_carveout: the cookbook holds out up to `eval_carve_ratio` of the rows (capped at
  `max_eval_seqs`) as an eval set, and it runs the loss eval after each epoch. These rows are
  removed from training, so the steps per epoch are computed from the remaining rows.
- A promotable checkpoint is saved at the end of each epoch (step = epoch * steps per epoch).
  The last one is the final checkpoint, which is promoted to `--output-model-id`.
- The trainer job id is written to `<log-path>/trainer_job.json` (RL patch) and to `sft_job.json`
  with the checkpoint-to-epoch map.
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
TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora"  # LORA_TRAINER, 32k context
# qwen3-4b-minimum is the full-parameter POLICY_TRAINER the RL run used; it does not take LoRA.
TRAIN_ON_WHAT = "all_assistant_messages"
SFT_JOB_FILE = "sft_job.json"


def count_rows(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def eval_rows(rows: int) -> int:
    """Rows the cookbook carves out for eval, with the cookbook's own defaults."""
    from recipes.sft_loop import DEFAULT_EVAL_CARVE_RATIO, DEFAULT_MAX_EVAL_SEQS, compute_eval_carveout

    return compute_eval_carveout(rows, DEFAULT_EVAL_CARVE_RATIO, DEFAULT_MAX_EVAL_SEQS)


def steps_per_epoch(train_rows: int, batch_size: int) -> int:
    return math.ceil(train_rows / batch_size)


def checkpoint_epochs(spe: int, epochs: int) -> dict[str, int]:
    """Checkpoint name -> epoch it ends, e.g. {"step-141": 1, "step-282": 2}."""
    return {f"step-{k * spe}": k for k in range(1, epochs + 1)}


def check_training_shape(shape: str, lora_rank: int, max_seq_len: int) -> None:
    """Fail before provisioning if the shape's trainer mode or context length does not fit the run."""
    import os

    from fireworks.training.sdk import TrainerJobManager

    manager = TrainerJobManager(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"),
    )
    profile = manager.resolve_training_profile(shape)
    want_mode = "LORA_TRAINER" if lora_rank > 0 else "POLICY_TRAINER"
    if profile.trainer_mode != want_mode:
        raise SystemExit(f"{shape} is {profile.trainer_mode}, but lora_rank={lora_rank} needs {want_mode}")
    if profile.max_supported_context_length < max_seq_len:
        raise SystemExit(
            f"{shape} supports {profile.max_supported_context_length} tokens, below max_seq_len={max_seq_len}"
        )
    print(f"Training shape ok: {shape} ({profile.trainer_mode}, {profile.max_supported_context_length} tokens)")


def build_config(args: argparse.Namespace, train_rows: int) -> Any:
    from recipes.sft_loop import Config
    from training.utils import TrainerConfig, WandBConfig

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
        eval_auto_carveout=True,
        sampler_save_interval=steps_per_epoch(train_rows, args.batch_size),
        save_final_checkpoint=True,  # promotes the last checkpoint to output_model_id
        output_model_id=args.output_model_id,
        trainer=TrainerConfig(training_shape_id=args.training_shape),
        wandb=WandBConfig(project="anli-sft"),
    )


def plan(args: argparse.Namespace) -> dict[str, Any]:
    rows = count_rows(args.dataset)
    n_eval = eval_rows(rows)
    train_rows = rows - n_eval
    spe = steps_per_epoch(train_rows, args.batch_size)
    return {
        "dataset": str(args.dataset),
        "rows": rows,
        "eval_rows": n_eval,
        "train_rows": train_rows,
        "steps_per_epoch": spe,
        "total_steps": spe * args.epochs,
        "checkpoints": checkpoint_epochs(spe, args.epochs),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "lora_rank": args.lora_rank,
        "renderer_name": RENDERER_NAME,
        "train_on_what": TRAIN_ON_WHAT,
        "output_model_id": args.output_model_id,
    }


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
    parser.add_argument("--output-model-id", default=None,
                        help="promote the final checkpoint to this model id (e.g. my-sft-model)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit without training")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = plan(args)
    print(json.dumps(summary, indent=2))
    print("Checkpoint -> epoch:")
    for name, epoch in summary["checkpoints"].items():
        print(f"  {name} = end of epoch {epoch}")
    if args.dry_run:
        return

    from anli_classification.rl_train import install_checkpoint_patches
    from recipes.sft_loop import main as run_sft

    check_training_shape(args.training_shape, args.lora_rank, args.max_seq_len)
    # Writes the trainer job id to <log-path>/trainer_job.json. promotable_checkpoints stays off: the
    # epoch-end saves are already promotable (sampler_due), so the patch's extra promotion does nothing here.
    install_checkpoint_patches(args.log_path, promotable_checkpoints=False)
    result = run_sft(build_config(args, summary["train_rows"]))
    job = {
        "job_id": result["job_id"],
        "steps": result["steps"],
        "output_model_id": args.output_model_id,
        "checkpoints": summary["checkpoints"],
        "train_rows": summary["train_rows"],
        "eval_rows": summary["eval_rows"],
    }
    args.log_path.mkdir(parents=True, exist_ok=True)
    (args.log_path / SFT_JOB_FILE).write_text(json.dumps(job, indent=2))
    print(json.dumps(job, indent=2))
    print(f"Wrote {args.log_path / SFT_JOB_FILE}")


if __name__ == "__main__":
    main()
