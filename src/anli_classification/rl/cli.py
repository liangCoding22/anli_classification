"""Command line and the run itself.

    anli-rl-train --pool-path <pool.jsonl> --training-shape <shape>            # the original defaults
    anli-rl-train --recipe v2 --pool-path <pool_v2.jsonl> --training-shape ... # the last run (see recipes.py)

Needs the Fireworks cookbook (Python 3.11+, Training API private preview); its imports are local to the functions that use
them, so this package still imports on Python 3.9.
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from anli_classification.config import DEFAULT_MODEL
from anli_classification.rl.cleanup import delete_run_resources, start_stall_watchdog
from anli_classification.rl.cookbook_patches import install_checkpoint_patches, install_hotload_timeout_patch
from anli_classification.rl.data import load_dev_rows, load_pool_rows
from anli_classification.rl.defaults import (
    ADVANTAGE_STD_FLOOR,
    CLASS_DROP_ALERT,
    COMPLETIONS_PER_PROMPT,
    DCP_SAVE_INTERVAL,
    DEV_ROWS,
    EPOCHS,
    EVAL_TEMPERATURE,
    JUDGE_MODEL,
    LEARNING_RATE,
    MALFORMED_MARGIN,
    MAX_COMPLETION_TOKENS,
    METRICS_FILE,
    PROMPT_GROUPS_PER_STEP,
    REWARD_VERSIONS,
    ROLLOUTS_FILE,
    SAMPLE_TIMEOUT_S,
    STALL_TIMEOUT_S,
    TEMPERATURE,
    WEIGHT_SYNC_TIMEOUT_S,
)
from anli_classification.rl.dev_metrics import dev_accuracy
from anli_classification.rl.guards import print_eval
from anli_classification.rl.launch import build_config, resolve_training_shape, write_run_config
from anli_classification.rl.recipes import RECIPES
from anli_classification.rl.logs import FormatTracker, RolloutLog
from anli_classification.rl.reward import ConsistencyJudge, floored_advantages
from anli_classification.rl.rollout import make_rollout_fn
from anli_classification.rl.stop import StopAtStep
from anli_classification.rl_summary import write_summary


def run_rl(args: argparse.Namespace) -> dict[str, Any]:
    from training.recipes.rl_loop import main
    from training.utils.rl.heldout import make_heldout_evaluation

    rows = load_pool_rows(Path(args.pool_path))
    if not rows:
        raise SystemExit(f"no examples in {args.pool_path}")
    if args.prompt_groups_per_step > len(rows):
        raise SystemExit(f"--prompt-groups-per-step {args.prompt_groups_per_step} exceeds {len(rows)} pool examples")
    args.training_shape = resolve_training_shape(args)
    config = build_config(args, len(rows))
    dev_rows = load_dev_rows(args.dev_rows, args.dev_seed)

    # Everything the run produces goes under log_path: run config, every cookbook metric
    # (train, rollout, sampler, perf, eval), and every graded rollout with its completion.
    log_dir = Path(args.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["COOKBOOK_METRICS_FILE"] = str(log_dir / METRICS_FILE)
    write_run_config(args, log_dir, rows, dev_rows)
    rollout_log = RolloutLog(log_dir / ROLLOUTS_FILE)
    format_tracker = FormatTracker(log_dir / METRICS_FILE, alert_rate=args.format_alert_rate)
    install_checkpoint_patches(log_dir, args.promotable_checkpoints)
    install_hotload_timeout_patch()
    eval_formats: dict[str, str] = {}  # answer format per dev row, filled by the eval rollouts
    eval_preds: dict[str, Optional[int]] = {}  # predicted label per dev row, filled by the eval rollouts
    consistency_judge = ConsistencyJudge(args.consistency_judge_model) if args.consistency_penalty > 0 else None

    evaluation_fn = print_eval(
        max_steps=args.max_steps,
        stop_filter_ratio=args.stop_filter_ratio,
        metrics_path=log_dir / METRICS_FILE,
        class_drop_alert=args.class_drop_alert,
        stop_class_drop=args.stop_class_drop,
        evaluation_fn=make_heldout_evaluation(
            dev_rows,
            samples_per_row=1,
            metrics_fn=functools.partial(dev_accuracy, formats=eval_formats, preds=eval_preds),
            max_concurrency=args.concurrency,
        )
    )
    started_at = time.time()
    start_stall_watchdog(args, log_dir, started_at)
    try:
        result = main(
            config,
            rollout_fn_factory=functools.partial(
                make_rollout_fn, rollout_log=rollout_log, format_tracker=format_tracker, eval_formats=eval_formats,
                eval_temperature=args.eval_temperature, eval_preds=eval_preds,
                consistency_judge=consistency_judge, consistency_penalty=args.consistency_penalty,
                reward_version=args.reward_version, malformed_margin=args.malformed_margin,
            ),
            rows=rows,
            evaluation_fn=evaluation_fn,
            evaluation_interval=args.eval_interval,
            **({"advantage_fn": functools.partial(floored_advantages, std_floor=args.advantage_std_floor)}
               if args.advantage_std_floor > 0 else {}),
        )
    except StopAtStep as stop:
        result = {"steps": stop.step}
        print(f"Stopped at step {stop.step} (--max-steps {args.max_steps})")
    except Exception:
        if args.delete_on_failure:
            try:
                for line in delete_run_resources(log_dir, started_at, args.base_model, deployment_id=args.deployment_id):
                    print(f"FAILURE CLEANUP: {line}", flush=True)
            except Exception as cleanup_error:
                print(f"FAILURE CLEANUP FAILED, delete the trainer and deployment yourself: {cleanup_error}", flush=True)
        raise
    format_tracker.flush()
    if consistency_judge is not None:
        print(consistency_judge.summary(), flush=True)
    # The cookbook saves the final checkpoint as step-<global_step> when any step was trained.
    final_checkpoint = f"step-{result['steps']}" if result.get("steps") else None
    write_summary(log_dir, final_checkpoint)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO post-training on the training pool (cookbook rl_loop).")
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="legacy",
                        help="a named set of flags (see recipes.py) used as the defaults; flags given explicitly "
                             "still win. legacy (default) changes nothing; v2 is the last run")
    parser.add_argument("--pool-path", required=True, help="pool.jsonl from anli-training-pool")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--training-shape", default=os.environ.get("TRAINING_SHAPE", ""))
    parser.add_argument("--policy-job-id", default=None, help="existing trainer job to reattach")
    parser.add_argument("--deployment-id", default=None, help="existing deployment to reattach")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="passes over the training rows; rejected rows also use up draws, so optimizer steps are fewer")
    parser.add_argument("--prompt-groups-per-step", type=int, default=PROMPT_GROUPS_PER_STEP,
                        help="prompts per optimizer step")
    parser.add_argument("--completions-per-prompt", type=int, default=COMPLETIONS_PER_PROMPT)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--eval-temperature", type=float, default=EVAL_TEMPERATURE,
                        help="sampling temperature for dev evaluation (training uses --temperature)")
    parser.add_argument("--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS)
    parser.add_argument("--lora-rank", type=int, default=0, help="0 = full-parameter")
    parser.add_argument("--checkpoint-interval", type=int, default=DCP_SAVE_INTERVAL)
    parser.add_argument(
        "--promotable-checkpoints", action=argparse.BooleanOptionalAction, default=True,
        help="also save each intermediate checkpoint as a promotable model (default on; --no-... to disable)",
    )
    parser.add_argument("--dev-rows", type=int, default=DEV_ROWS, help="fixed dev_r3 subset for evaluation")
    parser.add_argument("--dev-seed", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=5, help="evaluate on dev every N optimizer steps")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after this many optimizer steps (multiple of the checkpoint and eval intervals)")
    parser.add_argument("--stop-filter-ratio", type=float, default=None,
                        help="stop at an evaluation step when the last 3 steps' mean share of zero-variance "
                             "groups exceeds this (off by default)")
    parser.add_argument("--concurrency", type=int, default=32, help="max concurrent dev rollouts")
    parser.add_argument("--format-alert-rate", type=float, default=0.05,
                        help="alert when this share of recent training answers breaks the one-word format")
    parser.add_argument("--log-path", default="./anli-rl-logs")
    parser.add_argument("--output-model-id", default=None, help="promote the final checkpoint to this id")
    parser.add_argument("--skip-cleanup", action=argparse.BooleanOptionalAction, default=False,
                        help="keep trainer and deployment running (--no-skip-cleanup overrides a recipe)")
    parser.add_argument("--delete-on-failure", action=argparse.BooleanOptionalAction, default=True,
                        help="if the run crashes, delete its trainer job and deployment so they stop billing")
    parser.add_argument("--weight-sync-timeout", type=int, default=WEIGHT_SYNC_TIMEOUT_S,
                        help="seconds to wait for the deployment to load each new snapshot")
    parser.add_argument("--class-drop-alert", type=float, default=CLASS_DROP_ALERT,
                        help="print an ALERT when a class's dev recall falls this far below the first evaluation")
    parser.add_argument("--advantage-std-floor", type=float, default=ADVANTAGE_STD_FLOOR,
                        help="smallest group standard deviation GRPO divides by (0 = the cookbook's plain z-score)")
    parser.add_argument("--stop-class-drop", type=float, default=None,
                        help="stop at an evaluation where any class's dev recall is this far below the first evaluation")
    parser.add_argument("--consistency-penalty", type=float, default=0.0,
                        help="reward subtracted when an LLM judge finds the reasoning concluding a different label "
                             "than the output (training rollouts only; 0 = off)")
    parser.add_argument("--consistency-judge-model", default=JUDGE_MODEL)
    parser.add_argument("--reward-version", choices=REWARD_VERSIONS, default="v2",
                        help="v2 (default): the reward of the v2 run. v3: a malformed answer scores below every "
                             "parseable one, and an answer with no reasoning text is penalised like an inconsistent "
                             "one (only with --consistency-penalty > 0)")
    parser.add_argument("--malformed-margin", type=float, default=MALFORMED_MARGIN,
                        help="v3: how far below the worst parseable outcome a malformed answer scores")
    parser.add_argument("--sample-timeout", type=int, default=SAMPLE_TIMEOUT_S,
                        help="seconds before one sampling request is abandoned")
    parser.add_argument("--stall-timeout", type=int, default=STALL_TIMEOUT_S,
                        help="stop the run and delete its resources when no rollout or metric is written for this "
                             "many seconds (0 = off)")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "anli-rl"))
    # The recipe is read first (a full parse would fail on the missing required flags), then set as the defaults.
    chooser = argparse.ArgumentParser(add_help=False)
    chooser.add_argument("--recipe", choices=tuple(RECIPES), default="legacy")
    parser.set_defaults(**RECIPES[chooser.parse_known_args(argv)[0].recipe])
    args = parser.parse_args(argv)
    if args.completions_per_prompt < 2:
        parser.error("--completions-per-prompt must be at least 2 for GRPO")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.eval_interval < 1:
        parser.error("--eval-interval must be at least 1")
    if args.consistency_penalty < 0:
        parser.error("--consistency-penalty must be >= 0")
    if args.malformed_margin < 0:
        parser.error("--malformed-margin must be >= 0")
    if args.stop_filter_ratio is not None and not 0 < args.stop_filter_ratio < 1:
        parser.error("--stop-filter-ratio must be between 0 and 1")
    if args.max_steps is not None:
        if args.max_steps < 1:
            parser.error("--max-steps must be at least 1")
        if args.max_steps % args.checkpoint_interval or args.max_steps % args.eval_interval:
            parser.error("--max-steps must be a multiple of --checkpoint-interval and --eval-interval, "
                         "so the stop step is checkpointed and evaluated")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metrics = run_rl(args)
    print(f"RL training finished. Final metrics: {metrics}")
    if args.output_model_id:
        # Dev only: anli-eval defaults to dev and test, so the split is passed explicitly.
        # --template client: the promoted model's server template disables thinking (rendering.py).
        print(
            "Evaluate on dev (not test) once the model is deployed:\n"
            f"  anli-eval --model accounts/<account>/models/{args.output_model_id} "
            f"--splits dev --rounds r1 r2 r3 --template client --tokenizer-model {args.tokenizer_model}"
        )


if __name__ == "__main__":
    main()
