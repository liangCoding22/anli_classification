"""Turning command-line arguments into the cookbook's config, and recording what a run started with."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from anli_classification.rl.defaults import CONFIG_FILE, METRICS_FILE, RESULT_FILE, ROLLOUTS_FILE, TRAINER_JOB_FILE


def code_version() -> dict[str, Any]:
    """The git commit the run was started from and whether the tree had uncommitted changes (None outside a repository)."""
    def git(*command: str) -> Optional[str]:
        try:
            return subprocess.run(["git", *command], capture_output=True, text=True, check=True, cwd=Path(__file__).parent).stdout
        except Exception:
            return None

    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"git_sha": sha.strip() if sha else None, "dirty": bool(status.strip()) if status is not None else None}


def write_run_result(
    log_dir: Path,
    status: str,
    *,
    stop_reason: Optional[str] = None,
    final_step: Optional[int] = None,
    error: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> None:
    """Record how the run ended, for tools that would otherwise parse the log.

    status: "completed" (the loop finished), "stopped" (a guard or the step cap ended it; stop_reason says which),
    "failed" (an exception) or "stalled" (the watchdog ended it).
    """
    job_file = log_dir / TRAINER_JOB_FILE
    record = {
        "status": status,
        "stop_reason": stop_reason,
        "final_step": final_step,
        "error": error,
        "trainer_job_id": json.loads(job_file.read_text()).get("trainer_job_id") if job_file.exists() else None,
        "deployment_id": getattr(args, "deployment_id", None),
        "recipe": getattr(args, "recipe", None),
        "code": code_version(),
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / RESULT_FILE).write_text(json.dumps(record, indent=2))


def write_run_config(args: argparse.Namespace, log_dir: Path, train_rows: list[dict], dev_rows: list[dict]) -> None:
    """Record exactly what the run was started with, so a later debugging session can reproduce it."""
    record = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "code": code_version(),
        "train_keys": [row["key"] for row in train_rows],
        "dev_keys": [row["key"] for row in dev_rows],
        "cookbook_metrics_file": str(log_dir / METRICS_FILE),
        "rollouts_file": str(log_dir / ROLLOUTS_FILE),
    }
    (log_dir / CONFIG_FILE).write_text(json.dumps(record, indent=2))


def build_config(args: argparse.Namespace, n_rows: int) -> Any:
    from training.recipes.rl_loop import Config
    from training.utils import DeployConfig, TrainerConfig, WandBConfig

    return Config(
        log_path=args.log_path,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        completions_per_prompt=args.completions_per_prompt,
        prompt_groups_per_step=args.prompt_groups_per_step,
        epochs=args.epochs,
        max_rows=n_rows,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        lora_rank=args.lora_rank,
        dcp_save_interval=args.checkpoint_interval,
        cleanup_on_exit=not args.skip_cleanup,
        weight_sync_timeout=args.weight_sync_timeout,
        output_model_id=args.output_model_id,
        trainer=TrainerConfig(job_id=args.policy_job_id, training_shape_id=args.training_shape or None),
        deployment=DeployConfig(
            deployment_id=args.deployment_id,
            tokenizer_model=args.tokenizer_model,
            sample_timeout=args.sample_timeout,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.deployment_id or f"anli-rl-{int(time.time()) % 100000}",
        ),
    )


def resolve_training_shape(args: argparse.Namespace) -> str:
    """Use --training-shape if given; otherwise let the cookbook pick a validated shape for the base model.

    The deployment shape is derived from the training shape, so rl_loop fails without one.
    """
    if args.training_shape:
        return args.training_shape
    from fireworks.training.sdk.trainer import TrainerJobManager
    from training.utils import auto_select_training_shape

    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    manager = TrainerJobManager(api_key=os.environ["FIREWORKS_API_KEY"], base_url=base_url)
    shape = auto_select_training_shape(manager, base_model=args.base_model, lora_rank=args.lora_rank)
    print(f"Auto-selected training shape: {shape}")
    return shape
