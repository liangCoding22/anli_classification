"""GRPO post-training on the training pool, with the Fireworks cookbook's strict on-policy loop.

Input: `pool.jsonl` from `anli-training-pool` (one line per example, 8 sampled answers each).
Each example is a prompt. Every optimizer step samples `completions_per_prompt` answers per
prompt, rewards each answer 1.0 if its parsed label is gold and 0.0 otherwise, and trains on
the group-relative advantages. Groups where every answer gets the same reward carry no
advantage and are dropped by `has_reward_variance`.

The training loop lives in the Fireworks cookbook (`training/recipes/rl_loop.py`), which needs
Python >= 3.11, the Training API (private preview), and a training shape. Those imports are
kept inside the functions that use them, so this package still imports on Python 3.9.

Setup:
    git clone https://github.com/fw-ai/cookbook.git && cd cookbook/training && pip install -e .
    export PYTHONPATH=/path/to/cookbook:$PYTHONPATH   # the cookbook root holding `training/`
    export FIREWORKS_API_KEY=... TRAINING_SHAPE=...   # training shape from your account
    anli-rl-train --pool-path outputs/.../training_pool/<run>/pool.jsonl --training-shape "$TRAINING_SHAPE"
"""

from __future__ import annotations

import argparse
import functools
import math
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from anli_classification.parsers import LabelParser, is_well_formed, unparsed_reason
from anli_classification.rl_summary import write_summary
from anli_classification.prompts import PROMPTS, NliPromptV3
from anli_classification.schemas import Example
from anli_classification.config import DEFAULT_MODEL

# Settings from the cookbook's DeepMath example (qwen3-4b, 8 completions per prompt),
# adjusted for this task: sampling at temperature 1.0 for GRPO, and 4096 completion tokens so
# qwen3 reasoning is not cut off (the cookbook default of 1024 is too short for it).
FORMAT_SNAPSHOT_EVERY = 512  # training rollouts per format snapshot: 64 prompt groups x 8 completions
METRICS_FILE = "metrics.jsonl"  # every metric the cookbook logs, one JSON line per call
ROLLOUTS_FILE = "rollouts.jsonl"  # every graded rollout: key, gold, pred, reward, completion
CONFIG_FILE = "run_config.json"  # arguments and example keys the run started with
EPOCHS = 1  # passes over the training rows; rejected rows also use draws, so optimizer steps are fewer
COMPLETIONS_PER_PROMPT = 8  # rollouts per prompt (the GRPO group size)
PROMPT_GROUPS_PER_STEP = 64  # prompts per optimizer step
LEARNING_RATE = 1e-5
TEMPERATURE = 1.0
MAX_COMPLETION_TOKENS = 4096
DCP_SAVE_INTERVAL = 2  # resumable checkpoint every 2 optimizer steps

# Metrics to watch in the logs or W&B (names as the cookbook's async reference documents them;
# check them against the first step's logged keys in case the sync loop names them differently).
MONITOR_METRICS = (
    "rollout/raw_reward",  # mean reward before filtering; should rise
    "rollout/filtered_reward",  # mean reward on groups that survived the filter
    "rollout/filter_ratio",  # share of groups dropped for zero reward variance; should grow
    "rollout/truncated_ratio",  # completions hitting max tokens; should stay near 0
    "rollout/completion_tokens_mean",
    "train/ppo_kl",  # policy drift per step; a sudden jump means the learning rate is too high
    "train/ppo_clip_frac",
    "train/grad_norm",
)


def label_reward(completion: str, gold: int) -> float:
    """1.0 if the completion's first label word is the gold label, else 0.0."""
    return 1.0 if LabelParser().parse(completion) == gold else 0.0


def load_pool_rows(pool_path: Path, prompt_version: str = NliPromptV3.VERSION) -> list[dict[str, Any]]:
    """One training row per pool example: chat messages for the prompt, plus the gold label."""
    prompt = PROMPTS[prompt_version]()
    rows = []
    for line in pool_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        example = Example(
            split="train",
            round_id="r3",
            uid=record["key"].split("/", 1)[1],
            premise=record["premise"],
            hypothesis=record["hypothesis"],
            label=record["gold"],
            reason=record.get("reason", ""),
        )
        rows.append({"key": record["key"], "gold": record["gold"], "messages": prompt.build(example)})
    return rows


def load_dev_rows(n: int, seed: int, prompt_version: str = NliPromptV3.VERSION) -> list[dict[str, Any]]:
    """Fixed dev subset for intermediate evaluation: `n` random dev_r3 examples, same rows every time."""
    from anli_classification.data import AnliDataset

    prompt = PROMPTS[prompt_version]()
    examples = AnliDataset().load("dev", "r3", limit=n, seed=seed)
    return [{"key": ex.key, "gold": ex.label, "messages": prompt.build(ex)} for ex in examples]


def dev_accuracy(
    rows: Sequence[dict[str, Any]],
    runs_by_row: Sequence[Sequence[Any]],
    formats: Optional[dict[str, str]] = None,
) -> dict[str, float]:
    """Dev metrics. A run's reward is 1.0 when its answer is correct, 0.0 otherwise.

    Every expected rollout counts: a failed rollout (no run) and an invalid answer both count as
    incorrect, so neither can inflate the accuracy. `dev/accuracy` is the reward mean over all
    expected rollouts; `dev/valid_accuracy` is over the rollouts that returned an answer. `dev/strict_accuracy` is over the same rollouts and also needs a parsed answer.
    `formats` maps each row key to its parse status ('parsed' or an unparsed reason), recorded by the
    eval rollouts. It is cleared after each evaluation.
    """
    runs = [run for row_runs in runs_by_row for run in row_runs]
    scored = [run for run in runs if run is not None]
    correct = sum(run.segments[0].reward == 1.0 for run in scored)
    metrics = {
        "dev/accuracy": correct / len(runs) if runs else 0.0,
        "dev/valid_accuracy": correct / len(scored) if scored else 0.0,
        "dev/dropped_ratio": 1.0 - len(scored) / len(runs) if runs else 0.0,
        "dev/rows": float(len(rows)),
    }
    if formats is not None:
        strict = 0
        invalid = 0
        for row, row_runs in zip(rows, runs_by_row):
            parsed = formats.get(row["key"], "parsed") == "parsed"
            invalid += not parsed
            for run in row_runs:
                if run is not None and run.segments[0].reward == 1.0 and parsed:
                    strict += 1
        metrics["dev/strict_accuracy"] = strict / len(runs) if runs else 0.0
        metrics["dev/invalid_ratio"] = invalid / len(rows) if rows else 0.0
        metrics["dev/incorrect_ratio"] = (
            sum(1 for row, row_runs in zip(rows, runs_by_row)
                for run in row_runs
                if run is not None and run.segments[0].reward != 1.0
                and formats.get(row["key"], "parsed") == "parsed") / len(runs)
        ) if runs else 0.0
        formats.clear()
    return metrics


def thinking_text(message: Any) -> str:
    """The reasoning part of a sampled message (empty when the reply has no thinking part)."""
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return ""
    return "".join(part.get("thinking", "") for part in content if part.get("type") == "thinking")


class FormatTracker:
    """Counts malformed training answers, writes a snapshot per window, and alerts on a high rate.

    Snapshots go into metrics.jsonl (records have `type: format` and no `step`, so the cookbook's
    per-step records are unaffected).
    """

    def __init__(self, metrics_path: Path, window: int = FORMAT_SNAPSHOT_EVERY, alert_rate: float = 0.05) -> None:
        self.metrics_path = metrics_path
        self.window = window
        self.alert_rate = alert_rate
        self.total = 0
        self.window_total = 0
        self.window_malformed = 0
        self._lock = threading.Lock()

    def record(self, well_formed: bool) -> None:
        with self._lock:
            self.total += 1
            self.window_total += 1
            if not well_formed:
                self.window_malformed += 1
            if self.total % self.window == 0:
                self._flush_window()

    def flush(self) -> None:
        with self._lock:
            if self.window_total:
                self._flush_window()

    def _flush_window(self) -> None:  # caller holds the lock
        rate = self.window_malformed / self.window_total
        record = {
            "type": "format",
            "rollouts_total": self.total,
            "window_rollouts": self.window_total,
            "format/malformed": self.window_malformed,
            "format/malformed_rate": rate,
        }
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        if rate > self.alert_rate:
            print(
                f"ALERT: {rate:.1%} of the last {self.window_total} training answers are malformed "
                f"(not exactly one label word; threshold {self.alert_rate:.0%})",
                flush=True,
            )
        self.window_total = 0
        self.window_malformed = 0


class RolloutLog:
    """Appends one JSON line per graded rollout, from any thread."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record) + "\n"
        with self._lock, self.path.open("a") as fh:
            fh.write(line)


def make_rollout_fn(
    setup: Any,
    *,
    renderer_name: str = "",
    rollout_log: Optional[RolloutLog] = None,
    format_tracker: Optional[FormatTracker] = None,
    eval_formats: Optional[dict[str, str]] = None,
) -> Any:
    """Single-turn rollout: one sampled answer per call, graded by `label_reward`.

    `evaluation` is True for dev evaluation rollouts; each graded rollout is written to
    `rollout_log` with its completion text, so failures can be inspected later.
    Mirrors the cookbook's DeepMath example; the cookbook imports are local to this function.
    """
    import asyncio

    from training.renderer import get_text_content
    from training.utils import build_renderer, prepare_sampling_messages
    from training.utils.rl.rollout import single_turn_renderer_rollout

    renderer = build_renderer(setup.tokenizer, setup.tokenizer_id, renderer_name, load_image_processor=False)
    sampler = setup.sampler  # borrowed from the recipe; do not close

    async def rollout_fn(row: dict, evaluation: bool = False) -> Any:
        async def grade(row_: dict, message: Any, _parse_success: bool) -> float:
            completion = get_text_content(message)
            reward = await asyncio.to_thread(label_reward, completion, row_["gold"])
            well_formed = is_well_formed(completion)
            parse_status = unparsed_reason(completion) or "parsed"
            if format_tracker is not None and not evaluation:
                format_tracker.record(well_formed)
            if eval_formats is not None and evaluation:
                eval_formats[row_["key"]] = parse_status
            if rollout_log is not None:
                pred = LabelParser().parse(completion)
                thinking = thinking_text(message)
                rollout_log.write({
                    "split": "eval" if evaluation else "train",
                    "key": row_["key"],
                    "gold": row_["gold"],
                    "pred": pred,
                    "reward": reward,
                    "well_formed": well_formed,
                    "parse_status": parse_status,
                    "outcome": "correct" if reward == 1.0 else ("incorrect" if well_formed else "invalid"),
                    "completion": completion,
                    "thinking": thinking,
                    "thinking_chars": len(thinking),
                })
            return reward

        return await single_turn_renderer_rollout(
            row,
            renderer=renderer,
            sample_with_prompt_tokens=sampler.sample_with_prompt_tokens,
            message_builder=lambda r: prepare_sampling_messages(r.get("messages", [])),
            reward_fn=grade,
            sample_kwargs=setup.sample_kwargs,
            tokenizer=setup.tokenizer,
        )

    return rollout_fn


def write_run_config(args: argparse.Namespace, log_dir: Path, train_rows: list[dict], dev_rows: list[dict]) -> None:
    """Record exactly what the run was started with, so a later debugging session can reproduce it."""
    record = {
        "args": {key: str(value) for key, value in vars(args).items()},
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
        epochs=args.epochs if args.max_steps is None else epochs_for_steps(
            args.max_steps, args.prompt_groups_per_step, n_rows, args.epochs
        ),
        max_rows=n_rows,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        lora_rank=args.lora_rank,
        dcp_save_interval=args.checkpoint_interval,
        cleanup_on_exit=not args.skip_cleanup,
        output_model_id=args.output_model_id,
        trainer=TrainerConfig(job_id=args.policy_job_id, training_shape_id=args.training_shape or None),
        deployment=DeployConfig(
            deployment_id=args.deployment_id,
            tokenizer_model=args.tokenizer_model,
            sample_timeout=1200,
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


class StopAtStep(BaseException):
    """Ends training after the requested step. A BaseException, so the cookbook's eval wrapper does not catch it."""

    def __init__(self, step: int) -> None:
        super().__init__(f"stopped at step {step}")
        self.step = step


def print_eval(evaluation_fn: Any, max_steps: Optional[int] = None) -> Any:
    """Print each dev evaluation as a JSON line, and stop training at `max_steps`.

    The cookbook sends evaluation metrics only to W&B and to COOKBOOK_METRICS_FILE, so without
    a W&B login they would be dropped silently. The stop happens here because the eval hook runs
    after the step's checkpoint is saved; the cookbook's cleanup still runs on the way out.
    """

    async def wrapped(step: int, rollout_fn: Any) -> dict[str, float]:
        metrics = await evaluation_fn(step, rollout_fn)
        print(json.dumps({"type": "eval", "step": step, **metrics}), flush=True)
        if max_steps is not None and step >= max_steps:
            from training.utils.logging import log_metrics

            log_metrics(metrics, step=step)  # the cookbook would log this after the hook returns
            raise StopAtStep(step)
        return metrics

    return wrapped


TRAINER_JOB_FILE = "trainer_job.json"  # trainer job id, needed to list and promote checkpoints later
_INTERMEDIATE_NAME = re.compile(r"step-\d+")


def is_intermediate_checkpoint(name: str) -> bool:
    """Periodic checkpoints are named step-N; the final one is saved separately and already promotable."""
    return _INTERMEDIATE_NAME.fullmatch(name) is not None


HOTLOAD_REQUEST_TIMEOUT_S = 900  # the SDK's default is 60 s, too short for the first base-snapshot hotload


def install_hotload_timeout_patch() -> None:
    """Raise the default timeout of DeploymentManager.hotload, which hotload_and_wait never overrides.

    Without this, the first weight sync (a full base snapshot) times out after 60 s with a ReadTimeout
    and the run fails after its trainer and deployment are already running. The installed SDK is not modified.
    """
    from fireworks.training.sdk.deployment import DeploymentManager

    if getattr(DeploymentManager.hotload, "_anli_timeout_patched", False):
        return
    original = DeploymentManager.hotload

    @functools.wraps(original)
    def hotload(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", HOTLOAD_REQUEST_TIMEOUT_S)
        return original(self, *args, **kwargs)

    hotload._anli_timeout_patched = True
    DeploymentManager.hotload = hotload


def install_checkpoint_patches(log_dir: Path, promotable_checkpoints: bool) -> None:
    """Runtime patches on the cookbook's TrainingCheckpoints. The installed cookbook files are not modified.

    - Writes the trainer job id to `log_dir/trainer_job.json` when the trainer is created.
    - With `promotable_checkpoints`, each intermediate save (step-N) also saves a promotable
      base sampler snapshot, so it can be promoted to a model after the job finishes.
      The cookbook's periodic save passes promotable=False; this overrides only that.
    """
    from training.utils import checkpoints as cp

    if getattr(cp.TrainingCheckpoints, "_anli_patched", False):
        return
    original_init = cp.TrainingCheckpoints.__init__
    original_save = cp.TrainingCheckpoints.save

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        trainer_id = kwargs.get("trainer_id")
        if trainer_id:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / TRAINER_JOB_FILE).write_text(json.dumps({"trainer_job_id": trainer_id}, indent=2))

    def save(self: Any, name: str, *, resumable: bool, promotable: bool, data_consumed: Any = None) -> None:
        if promotable_checkpoints and resumable and not promotable and is_intermediate_checkpoint(name):
            promotable = True
        return original_save(self, name, resumable=resumable, promotable=promotable, data_consumed=data_consumed)

    cp.TrainingCheckpoints.__init__ = init
    cp.TrainingCheckpoints.save = save
    cp.TrainingCheckpoints._anli_patched = True


def epochs_for_steps(max_steps: int, prompt_groups_per_step: int, n_rows: int, epochs: int) -> int:
    """Passes over the rows that leave room for `max_steps` optimizer steps, with a 3x margin.

    Groups rejected by the zero-variance filter also use row draws, so the margin covers them.
    """
    needed = math.ceil(3 * max_steps * prompt_groups_per_step / n_rows)
    return max(epochs, needed)


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

    evaluation_fn = print_eval(
        max_steps=args.max_steps,
        evaluation_fn=make_heldout_evaluation(
            dev_rows,
            samples_per_row=1,
            metrics_fn=functools.partial(dev_accuracy, formats=eval_formats),
            max_concurrency=args.concurrency,
        )
    )
    try:
        result = main(
            config,
            rollout_fn_factory=functools.partial(
                make_rollout_fn, rollout_log=rollout_log, format_tracker=format_tracker, eval_formats=eval_formats
            ),
            rows=rows,
            evaluation_fn=evaluation_fn,
            evaluation_interval=args.eval_interval,
        )
    except StopAtStep as stop:
        result = {"steps": stop.step}
        print(f"Stopped at step {stop.step} (--max-steps {args.max_steps})")
    format_tracker.flush()
    # The cookbook saves the final checkpoint as step-<global_step> when any step was trained.
    final_checkpoint = f"step-{result['steps']}" if result.get("steps") else None
    write_summary(log_dir, final_checkpoint)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO post-training on the training pool (cookbook rl_loop).")
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
    parser.add_argument("--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS)
    parser.add_argument("--lora-rank", type=int, default=0, help="0 = full-parameter")
    parser.add_argument("--checkpoint-interval", type=int, default=DCP_SAVE_INTERVAL)
    parser.add_argument(
        "--promotable-checkpoints", action=argparse.BooleanOptionalAction, default=True,
        help="also save each intermediate checkpoint as a promotable model (default on; --no-... to disable)",
    )
    parser.add_argument("--dev-rows", type=int, default=200, help="fixed dev_r3 subset for evaluation")
    parser.add_argument("--dev-seed", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=5, help="evaluate on dev every N optimizer steps")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after this many optimizer steps (multiple of the checkpoint and eval intervals)")
    parser.add_argument("--concurrency", type=int, default=32, help="max concurrent dev rollouts")
    parser.add_argument("--format-alert-rate", type=float, default=0.05,
                        help="alert when this share of recent training answers breaks the one-word format")
    parser.add_argument("--log-path", default="./anli-rl-logs")
    parser.add_argument("--output-model-id", default=None, help="promote the final checkpoint to this id")
    parser.add_argument("--skip-cleanup", action="store_true", help="keep trainer and deployment running")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "anli-rl"))
    args = parser.parse_args(argv)
    if args.completions_per_prompt < 2:
        parser.error("--completions-per-prompt must be at least 2 for GRPO")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.eval_interval < 1:
        parser.error("--eval-interval must be at least 1")
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
        print(
            "Evaluate on dev (not test) once the model is deployed:\n"
            f"  anli-eval --model accounts/<account>/models/{args.output_model_id} "
            "--splits dev --rounds r1 r2 r3"
        )


if __name__ == "__main__":
    main()
