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
import json
import os
import re
import threading
import time
from collections import Counter
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from anli_classification.parsers import LabelParser, is_well_formed, unparsed_reason
from anli_classification.rl_summary import write_summary
from anli_classification.prompts import PROMPTS, NliPromptV3
from anli_classification.schemas import LABELS, Example
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
EVAL_TEMPERATURE = 0.0  # dev evaluation is greedy; the cookbook would reuse the training temperature
MAX_COMPLETION_TOKENS = 4096
DEV_ROWS = 1000  # about 330 per class, so per-class recall has a standard error near 2.6 points
JUDGE_MODEL = "accounts/fireworks/models/gpt-oss-120b"  # serverless; tested on 120 real rollouts, 4 flagged, all genuine
JUDGE_MAX_CHARS = 16000  # reasoning longer than this keeps its first 3000 and last 12000 characters
REWARD_VERSIONS = ("v2", "v3")  # v2: the reward of the v2 run (kept as the default); v3: see training_reward
MALFORMED_MARGIN = 0.25  # v3: a malformed answer scores this far below the worst parseable outcome
NO_REASONING = "no_reasoning"  # verdict recorded for a parseable answer that has no reasoning text (v3)
ADVANTAGE_STD_FLOOR = 0.2  # binary groups with >= 1 success in 16 have std >= 0.25, so only penalty-only groups are affected
CLASS_DROP_ALERT = 0.05  # alert when a class's dev recall falls this far below the run's first evaluation
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
    """1.0 if the completion's final answer is exactly one label word and it is the gold label, else 0.0."""
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
        rows.append({"key": record["key"], "gold": record["gold"], "messages": prompt.build(example),
                     "premise": record["premise"], "hypothesis": record["hypothesis"]})
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
    preds: Optional[dict[str, Optional[int]]] = None,
) -> dict[str, float]:
    """Dev metrics. A run's reward is 1.0 when its answer is correct, 0.0 otherwise.

    Every expected rollout counts: a failed rollout (no run) and an invalid answer both count as
    incorrect, so neither can inflate the accuracy. `dev/accuracy` is the reward mean over all
    expected rollouts; `dev/valid_accuracy` is over the rollouts that returned an answer. `dev/strict_accuracy` is over the same rollouts and also needs a parsed answer.
    `formats` maps each row key to its parse status ('parsed' or an unparsed reason), recorded by the
    eval rollouts. It is cleared after each evaluation.

    `preds` maps each row key to the label the eval rollout predicted (None when unparsed). With it, the
    metrics also hold per-class recall, each class's share of predictions, and macro-F1, so a class
    trade-off that leaves `dev/accuracy` unchanged is visible. A failed rollout counts as incorrect
    and as no prediction. It is cleared after each evaluation.
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
    if preds is not None:
        metrics.update(class_metrics(rows, runs_by_row, preds))
        preds.clear()
    return metrics


def class_metrics(
    rows: Sequence[dict[str, Any]],
    runs_by_row: Sequence[Sequence[Any]],
    preds: dict[str, Optional[int]],
) -> dict[str, float]:
    """Per-class recall, predicted share and macro-F1 over the dev rows (one rollout per row)."""
    n_classes = len(LABELS)
    gold_count = [0] * n_classes
    hits = [0] * n_classes
    predicted = [0] * n_classes
    for row, row_runs in zip(rows, runs_by_row):
        gold = row["gold"]
        gold_count[gold] += 1
        run = row_runs[0] if row_runs else None
        pred = preds.get(row["key"]) if run is not None else None
        if pred is not None:
            predicted[pred] += 1
            hits[gold] += pred == gold
    total = sum(gold_count)
    metrics: dict[str, float] = {}
    f1s = []
    for i, label in enumerate(LABELS):
        recall = hits[i] / gold_count[i] if gold_count[i] else 0.0
        precision = hits[i] / predicted[i] if predicted[i] else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        metrics[f"dev/recall_{label}"] = recall
        metrics[f"dev/predicted_share_{label}"] = predicted[i] / total if total else 0.0
    metrics["dev/macro_f1"] = sum(f1s) / n_classes
    return metrics


JUDGE_SYSTEM = (
    "You check a model's reasoning about a natural language inference problem. You are given the premise, the "
    "hypothesis, the model's full reasoning, and the label the model finally output. Decide whether the reasoning "
    "actually arrives at that label. Reply with one line: '<verdict> <label>' where verdict is consistent (the "
    "reasoning concludes the output label), inconsistent (the reasoning concludes a different label), or unclear "
    "(it never settles on one label), and label is the label the reasoning itself concludes (entailment, neutral, "
    "contradiction) or none. Do not judge whether the label is correct."
)
_JUDGE_VERDICT = re.compile(r"^\W*(inconsistent|consistent|unclear)\b", re.IGNORECASE)


def judge_prompt(premise: str, hypothesis: str, thinking: str, label: str) -> str:
    if len(thinking) > JUDGE_MAX_CHARS:
        thinking = thinking[:3000] + "\n[...]\n" + thinking[-12000:]
    return (f"Premise:\n{premise}\n\nHypothesis:\n{hypothesis}\n\nModel's reasoning:\n{thinking}\n\n"
            f"Label the model output: {label}")


def parse_judge_reply(text: Optional[str]) -> Optional[str]:
    """'consistent', 'inconsistent' or 'unclear' from the judge's first word; None when it says anything else.

    Only the verdict is used. The label the judge adds is logged by nobody: on one test case it contradicted its own verdict.
    """
    match = _JUDGE_VERDICT.match(text or "")
    return match.group(1).lower() if match else None


def floored_advantages(rewards: Sequence[float], std_floor: float = ADVANTAGE_STD_FLOOR, eps: float = 1e-8) -> list[float]:
    """The cookbook's group-normalised advantages, but never dividing by a spread below `std_floor`.

    GRPO divides each reward's distance from the group mean by the group's standard deviation. When a small
    reward term (the consistency penalty) is the only difference inside a group, that division stretches it to
    the size of a full right-versus-wrong difference. The floor keeps such a term in proportion to its size.
    Groups with equal rewards still get zero advantage, and `std_floor` 0 reproduces the cookbook.
    """
    n = len(rewards)
    mean = sum(rewards) / n
    std = (sum((r - mean) ** 2 for r in rewards) / (n - 1)) ** 0.5 if n > 1 else 0.0
    if std < eps:
        return [0.0] * n
    return [(r - mean) / (max(std, std_floor) + eps) for r in rewards]


def shaped_reward(label_reward_value: float, verdict: Optional[str], penalty: float) -> float:
    """The label reward, minus `penalty` when the judge found the reasoning concluding a different label.

    The term differs between samples of one prompt, so it survives the cookbook's per-group z-scoring.
    """
    return label_reward_value - (penalty if verdict == "inconsistent" else 0.0)


def training_reward(
    label_value: float,
    pred: Optional[int],
    verdict: Optional[str],
    penalty: float,
    version: str = "v2",
    malformed_margin: float = MALFORMED_MARGIN,
) -> float:
    """The reward a training rollout gets: the label reward, adjusted for the consistency judge.

    v2 (default, the reward of the v2 run): `label_value - penalty` when the verdict is "inconsistent". Two gaps: a
    malformed answer (no judge call) scores 0, which is more than a wrong answer whose reasoning contradicts its
    label (-penalty), and an answer with no reasoning text skips the judge, so it can never be penalised.

    v3 (with penalty > 0) closes both, keeping the order
    correct > wrong > wrong and inconsistent > malformed:
      - malformed (no parseable label): -(penalty + malformed_margin), below every parseable outcome;
      - parseable but no reasoning text (verdict NO_REASONING): `label_value - penalty`, as if inconsistent.
    With penalty 0 every version is the plain label reward.
    """
    if version == "v3" and penalty > 0:
        if pred is None:
            return -(penalty + malformed_margin)
        if verdict == NO_REASONING:
            return label_value - penalty
    return shaped_reward(label_value, verdict, penalty)


class ConsistencyJudge:
    """Asks an LLM whether a rollout's reasoning arrives at the label it output. Fails open.

    The judge sees the premise, hypothesis, full reasoning and the model's output label, never the gold label.
    Any error, timeout or unparseable reply gives no verdict (so no penalty) and is counted, so a judge outage
    cannot stop or distort training. Replies are cached by input.
    """

    def __init__(self, model: str = JUDGE_MODEL, post: Any = None, report_every: int = 500) -> None:
        self.model = model
        self.post = post or self._post
        self.report_every = report_every
        self.stats: Counter = Counter()
        self._cache: dict[str, Optional[str]] = {}
        self._lock = threading.Lock()

    def _post(self, body: dict) -> dict:
        base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
        request = urllib.request.Request(
            f"{base_url}/inference/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())

    def verdict(self, premise: str, hypothesis: str, thinking: str, label: str) -> Optional[str]:
        prompt = judge_prompt(premise, hypothesis, thinking, label)
        with self._lock:
            if prompt in self._cache:
                self.stats["cache_hits"] += 1
                return self._cache[prompt]
        body = {"model": self.model, "temperature": 0, "max_tokens": 300, "reasoning_effort": "low",
                "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}]}
        result: Optional[str] = None
        for attempt in range(3):
            try:
                reply = self.post(body)["choices"][0]["message"].get("content")
                result = parse_judge_reply(reply)
                if result is None:
                    self.stats["unparsed"] += 1
                break
            except Exception:
                if attempt == 2:
                    self.stats["errors"] += 1
                else:
                    time.sleep(1.5 * (attempt + 1))
        with self._lock:
            self._cache[prompt] = result
            self.stats["calls"] += 1
            self.stats[result or "none"] += 1
            if self.stats["calls"] % self.report_every == 0:
                print(self.summary(), flush=True)
        return result

    def record(self, name: str) -> None:
        """Count an event that did not need a call (e.g. a rollout with no reasoning to judge)."""
        with self._lock:
            self.stats[name] += 1

    def summary(self) -> str:
        s = self.stats
        extra = f", no reasoning {s[NO_REASONING]}" if s[NO_REASONING] else ""
        return (f"JUDGE: {s['calls']} calls, inconsistent {s['inconsistent']}, unclear {s['unclear']}, "
                f"errors {s['errors']}, unparsed {s['unparsed']}{extra}")


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
    eval_temperature: Optional[float] = EVAL_TEMPERATURE,
    eval_preds: Optional[dict[str, Optional[int]]] = None,
    consistency_judge: Optional[ConsistencyJudge] = None,
    consistency_penalty: float = 0.0,
    reward_version: str = "v2",
    malformed_margin: float = MALFORMED_MARGIN,
) -> Any:
    """Single-turn rollout: one sampled answer per call, graded by `label_reward`.

    `evaluation` is True for dev evaluation rollouts, which sample at `eval_temperature` (None keeps the
    training temperature); each graded rollout is written to
    `rollout_log` with its completion text, so failures can be inspected later.
    Mirrors the cookbook's DeepMath example; the cookbook imports are local to this function.
    """
    import asyncio

    from training.renderer import get_text_content
    from training.utils import build_renderer, prepare_sampling_messages
    from training.utils.rl.rollout import single_turn_renderer_rollout

    renderer = build_renderer(setup.tokenizer, setup.tokenizer_id, renderer_name, load_image_processor=False)
    sampler = setup.sampler  # borrowed from the recipe; do not close
    eval_sample_kwargs = (
        setup.sample_kwargs if eval_temperature is None else {**setup.sample_kwargs, "temperature": eval_temperature}
    )

    async def rollout_fn(row: dict, evaluation: bool = False) -> Any:
        async def grade(row_: dict, message: Any, _parse_success: bool) -> float:
            completion = get_text_content(message)
            label_value = await asyncio.to_thread(label_reward, completion, row_["gold"])
            pred = LabelParser().parse(completion)
            thinking = thinking_text(message)
            verdict = None
            if consistency_judge is not None and not evaluation and pred is not None and thinking:
                verdict = await asyncio.to_thread(
                    consistency_judge.verdict, row_.get("premise", ""), row_.get("hypothesis", ""), thinking, LABELS[pred]
                )
            if (not evaluation and consistency_judge is not None and reward_version == "v3"
                    and consistency_penalty > 0 and pred is not None and not thinking):
                verdict = NO_REASONING
                consistency_judge.record(NO_REASONING)
            # evaluation never gets a judge or a penalty: it is the plain label reward
            reward = training_reward(label_value, pred, verdict, 0.0 if evaluation else consistency_penalty,
                                     reward_version, malformed_margin)
            well_formed = is_well_formed(completion)
            parse_status = unparsed_reason(completion) or "parsed"
            if format_tracker is not None and not evaluation:
                format_tracker.record(well_formed)
            if eval_formats is not None and evaluation:
                eval_formats[row_["key"]] = parse_status
            if eval_preds is not None and evaluation:
                eval_preds[row_["key"]] = pred
            if rollout_log is not None:
                rollout_log.write({
                    "split": "eval" if evaluation else "train",
                    "key": row_["key"],
                    "gold": row_["gold"],
                    "pred": pred,
                    "reward": reward,
                    "label_reward": label_value,
                    "judge_verdict": verdict,
                    "well_formed": well_formed,
                    "parse_status": parse_status,
                    "outcome": "correct" if label_value == 1.0 else ("incorrect" if well_formed else "invalid"),
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
            sample_kwargs=eval_sample_kwargs if evaluation else setup.sample_kwargs,
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


class StopAtStep(BaseException):
    """Ends training after the requested step. A BaseException, so the cookbook's eval wrapper does not catch it."""

    def __init__(self, step: int) -> None:
        super().__init__(f"stopped at step {step}")
        self.step = step


def recent_filter_ratio(metrics_path: Path, window: int) -> Optional[float]:
    """Mean `rollout/filter_ratio` over the last `window` logged steps, or None until `window` steps exist."""
    if not metrics_path.exists():
        return None
    ratios = []
    for line in metrics_path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if "rollout/filter_ratio" in record:
                ratios.append(record["rollout/filter_ratio"])
    if len(ratios) < window:
        return None
    return sum(ratios[-window:]) / window


def print_eval(
    evaluation_fn: Any,
    max_steps: Optional[int] = None,
    stop_filter_ratio: Optional[float] = None,
    metrics_path: Optional[Path] = None,
    filter_window: int = 3,
    class_drop_alert: Optional[float] = None,
    stop_class_drop: Optional[float] = None,
) -> Any:
    """Print each dev evaluation as a JSON line, and stop training at `max_steps`.

    With `stop_filter_ratio`, it also stops once the mean filter ratio of the last `filter_window` steps
    exceeds it: most groups then have no reward variance, so further steps mostly pay for dropped rollouts.
    The check runs at evaluation steps only, so set `--eval-interval` to match `--checkpoint-interval`
    to have the stop step saved.

    With `class_drop_alert`, it prints an ALERT when a class's dev recall falls that far below the run's
    first evaluation (alert only). With `stop_class_drop`, a drop that large also stops the run, so the last
    checkpoint before a class collapses is the one kept.

    The cookbook sends evaluation metrics only to W&B and to COOKBOOK_METRICS_FILE, so without
    a W&B login they would be dropped silently. The stop happens here because the eval hook runs
    after the step's checkpoint is saved; the cookbook's cleanup still runs on the way out.
    """

    baseline: dict[str, float] = {}

    async def wrapped(step: int, rollout_fn: Any) -> dict[str, float]:
        metrics = await evaluation_fn(step, rollout_fn)
        print(json.dumps({"type": "eval", "step": step, **metrics}), flush=True)
        class_stop = False
        if class_drop_alert is not None or stop_class_drop is not None:
            for label in LABELS:
                key = f"dev/recall_{label}"
                if key not in metrics:
                    continue
                baseline.setdefault(key, metrics[key])
                drop = baseline[key] - metrics[key]
                if class_drop_alert is not None and drop > class_drop_alert:
                    print(f"ALERT: {label} dev recall {metrics[key]:.3f} is {drop:.3f} below the first evaluation "
                          f"({baseline[key]:.3f}) at step {step}", flush=True)
                if stop_class_drop is not None and drop > stop_class_drop:
                    print(f"Stopping at step {step}: {label} dev recall fell {drop:.3f} below the first evaluation "
                          f"(limit {stop_class_drop:.3f})", flush=True)
                    class_stop = True
        stop = class_stop or (max_steps is not None and step >= max_steps)
        if not stop and stop_filter_ratio is not None and metrics_path is not None:
            ratio = recent_filter_ratio(metrics_path, filter_window)
            if ratio is not None and ratio > stop_filter_ratio:
                print(f"Stopping at step {step}: filter ratio {ratio:.0%} over the last {filter_window} steps "
                      f"exceeds {stop_filter_ratio:.0%}", flush=True)
                stop = True
        if stop:
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


SAMPLE_TIMEOUT_S = 300  # per sampling request; a hung connection should fail in minutes, not 20
STALL_TIMEOUT_S = 1200  # kill the run when neither rollouts.jsonl nor metrics.jsonl changes for this long
WEIGHT_SYNC_TIMEOUT_S = 1800  # how long the cookbook waits for the deployment to report a new snapshot (default 600)
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


def _fireworks_api(method: str, path: str, api_key: str, base_url: str) -> Any:
    request = urllib.request.Request(f"{base_url}/v1/{path}", method=method, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    return json.loads(body) if body else {}


def delete_run_resources(
    log_dir: Path, started_at: float, base_model: str, api: Any = _fireworks_api, deployment_id: Optional[str] = None
) -> list[str]:
    """Delete the trainer job and the deployment this run created, so a crashed run stops billing.

    The cookbook only scales the deployment to zero on exit, and does nothing under --skip-cleanup. The trainer
    comes from `trainer_job.json` (or, when that was never written, any trainer on `base_model` created after
    `started_at`); deployments are those on `base_model` created after `started_at`,
    plus `deployment_id` when the run reattached to an older one.
    Every step is best effort and reported, so a failed delete never hides the original error.
    """
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    call = functools.partial(api, api_key=api_key, base_url=base_url)
    since = datetime.fromtimestamp(started_at, timezone.utc) - timedelta(minutes=2)
    done: list[str] = []

    def delete(path: str) -> None:
        try:
            call("DELETE", path)
            done.append(f"deleted {path}")
        except Exception as exc:  # already gone or refused: report and carry on
            done.append(f"could not delete {path}: {exc}")

    account = call("GET", "accounts")["accounts"][0]["name"]
    job_file = log_dir / TRAINER_JOB_FILE
    if job_file.exists():
        delete(f"{account}/rlorTrainerJobs/{json.loads(job_file.read_text())['trainer_job_id']}")
    else:  # the file is written once the trainer is up, so a run that never got a GPU has no record of its job
        for job in call("GET", f"{account}/rlorTrainerJobs?pageSize=100").get("rlorTrainerJobs", []):
            created = datetime.fromisoformat(job["createTime"].replace("Z", "+00:00"))
            if (job.get("state") != "JOB_STATE_DELETED" and created >= since
                    and job.get("trainingConfig", {}).get("baseModel") == base_model):
                delete(job["name"])
    if deployment_id:
        delete(f"{account}/deployments/{deployment_id}?ignoreChecks=true")
    for deployment in call("GET", f"{account}/deployments?pageSize=100").get("deployments", []):
        if deployment_id and deployment["name"].endswith(f"/{deployment_id}"):
            continue
        created = datetime.fromisoformat(deployment["createTime"].replace("Z", "+00:00"))
        if deployment.get("baseModel") == base_model and created >= since:
            delete(f"{deployment['name']}?ignoreChecks=true")
    return done


class StallWatchdog:
    """Detects a run that has stopped making progress (e.g. hung connections) by watching its output files."""

    def __init__(self, paths: Sequence[Path], timeout: float, clock: Any = time.time) -> None:
        self.paths = list(paths)
        self.timeout = timeout
        self.clock = clock
        self.started = clock()

    def last_activity(self) -> float:
        mtimes = [p.stat().st_mtime for p in self.paths if p.exists()]
        return max([self.started, *mtimes])

    def stalled(self) -> bool:
        return self.clock() - self.last_activity() > self.timeout

    def watch(self, on_stall: Any, interval: float = 30.0) -> None:
        while not self.stalled():
            time.sleep(interval)
        on_stall()


def start_stall_watchdog(args: argparse.Namespace, log_dir: Path, started_at: float) -> None:
    """Background thread: on a stall, delete the run's cloud resources and end the process."""
    if not args.stall_timeout:
        return

    def on_stall() -> None:
        print(f"STALL: no rollouts or metrics for {args.stall_timeout}s; stopping the run", flush=True)
        if args.delete_on_failure:
            try:
                for line in delete_run_resources(log_dir, started_at, args.base_model, deployment_id=args.deployment_id):
                    print(f"STALL CLEANUP: {line}", flush=True)
            except Exception as error:
                print(f"STALL CLEANUP FAILED, delete the trainer and deployment yourself: {error}", flush=True)
        os._exit(1)  # the main thread is stuck inside the cookbook loop, so it cannot be interrupted cleanly

    watchdog = StallWatchdog([log_dir / ROLLOUTS_FILE, log_dir / METRICS_FILE], args.stall_timeout)
    threading.Thread(target=watchdog.watch, args=(on_stall,), daemon=True).start()


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
    parser.add_argument("--skip-cleanup", action="store_true", help="keep trainer and deployment running")
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
