"""Post-run summary of an RL run, built only from the files the run wrote (no API calls).

Reads `run_config.json`, `metrics.jsonl` and `rollouts.jsonl` from the run's log folder, and
writes `summary.txt`. Every value is either computed from those files or reported as
"unavailable". Nothing is estimated silently.

Definitions (from the cookbook's metric code):
- An optimizer batch logs `rollout/raw_samples` (every sampled completion, before the
  zero-variance filter) and `rollout/filtered_samples` (completions in groups that were trained).
  Both count completions, so groups = completions / completions_per_prompt.
- Acceptance rate = accepted groups / attempted groups.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Optional, Sequence

UNAVAILABLE = "unavailable"
CONFIG_FILE = "run_config.json"
METRICS_FILE = "metrics.jsonl"
ROLLOUTS_FILE = "rollouts.jsonl"
SUMMARY_FILE = "summary.txt"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def weighted_mean(pairs: Sequence[tuple[float, float]]) -> Optional[float]:
    total = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total if total else None




def format_counts(rollouts: Sequence[dict[str, Any]], split: str) -> Optional[dict[str, Any]]:
    rows = [r for r in rollouts if r.get("split") == split]
    if not rows:
        return None
    malformed = sum(1 for r in rows if not r.get("well_formed", False))
    statuses = collections.Counter(r.get("parse_status", "unknown") for r in rows)
    return {
        "rollouts": len(rows),
        "well_formed": len(rows) - malformed,
        "malformed": malformed,
        "malformed_rate": malformed / len(rows),
        "parse_status": dict(statuses),
        "outcome": dict(collections.Counter(r.get("outcome", "unknown") for r in rows)),
    }

def summarize(log_dir: Path, final_checkpoint: Optional[str] = None) -> dict[str, Any]:
    config = json.loads((log_dir / CONFIG_FILE).read_text())
    args = config["args"]
    group_size = int(args["completions_per_prompt"])
    metrics = read_jsonl(log_dir / METRICS_FILE)
    rollouts = read_jsonl(log_dir / ROLLOUTS_FILE)

    batches = [m for m in metrics if "rollout/raw_samples" in m]
    raw = sum(m["rollout/raw_samples"] for m in batches)
    kept = sum(m["rollout/filtered_samples"] for m in batches)
    attempted_groups = round(raw / group_size) if batches else None
    accepted_groups = round(kept / group_size) if batches else None

    evals = sorted((m for m in metrics if "dev/accuracy" in m), key=lambda m: m["step"])
    token_pairs = [(m["rollout/completion_tokens_mean"], m["rollout/filtered_samples"])
                   for m in batches if "rollout/completion_tokens_mean" in m]
    trunc_pairs = [(m["rollout/truncated_ratio"], m["rollout/filtered_samples"])
                   for m in batches if "rollout/truncated_ratio" in m]

    summary: dict[str, Any] = {
        "unique_training_examples": len(config.get("train_keys", [])) or UNAVAILABLE,
        "epochs_planned": args.get("epochs", UNAVAILABLE),
        "attempted_groups": attempted_groups if batches else UNAVAILABLE,
        "accepted_groups": accepted_groups if batches else UNAVAILABLE,
        "rejected_all_correct": UNAVAILABLE,  # the cookbook logs only the total rejected, not by bucket
        "rejected_all_wrong": UNAVAILABLE,
        "acceptance_rate": (accepted_groups / attempted_groups) if batches and attempted_groups else UNAVAILABLE,
        "optimizer_updates": len(batches) if batches else UNAVAILABLE,
        "trained_completion_tokens_approx": (
            sum(mean * n for mean, n in token_pairs) if token_pairs else UNAVAILABLE
        ),
        "average_completion_tokens": weighted_mean(token_pairs) if token_pairs else UNAVAILABLE,
        "truncation_rate": weighted_mean(trunc_pairs) if trunc_pairs else UNAVAILABLE,
        "train_rollouts": format_counts(rollouts, "train") or UNAVAILABLE,
        "eval_rollouts": format_counts(rollouts, "eval") or UNAVAILABLE,
        "format_snapshots": len([m for m in metrics if m.get("type") == "format"]),
        "initial_dev_accuracy": (
            {"step": evals[0]["step"], "accuracy": evals[0]["dev/accuracy"]} if evals else UNAVAILABLE
        ),
        "best_dev_accuracy": (
            max(({"step": m["step"], "accuracy": m["dev/accuracy"]} for m in evals),
                key=lambda d: d["accuracy"]) if evals else UNAVAILABLE
        ),
        "final_dev_accuracy": (
            {"step": evals[-1]["step"], "accuracy": evals[-1]["dev/accuracy"]} if evals else UNAVAILABLE
        ),
        "dev_evaluations": [{"step": m["step"], "accuracy": m["dev/accuracy"],
                             "valid_accuracy": m.get("dev/valid_accuracy", UNAVAILABLE),
                             "strict_accuracy": m.get("dev/strict_accuracy", UNAVAILABLE),
                             "invalid_ratio": m.get("dev/invalid_ratio", UNAVAILABLE),
                             "incorrect_ratio": m.get("dev/incorrect_ratio", UNAVAILABLE)} for m in evals],
        "final_checkpoint": final_checkpoint or UNAVAILABLE,
    }
    summary["text"] = format_summary(summary)
    return summary


def _fmt(value: Any) -> str:
    if value == UNAVAILABLE or value is None:
        return UNAVAILABLE
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        if "accuracy" in value:
            return f"{value['accuracy']:.4f} (step {value['step']})"
        return ", ".join(f"{k} {v}" for k, v in value.items())
    return str(value)


def format_summary(s: dict[str, Any]) -> str:
    rate = s["acceptance_rate"]
    lines = [
        "Training Summary",
        "----------------",
        f"Unique training examples: {_fmt(s['unique_training_examples'])}",
        f"Epochs (planned): {_fmt(s['epochs_planned'])}",
        f"Attempted groups: {_fmt(s['attempted_groups'])}",
        f"Accepted groups: {_fmt(s['accepted_groups'])}",
        f"Rejected all-correct: {_fmt(s['rejected_all_correct'])}",
        f"Rejected all-wrong: {_fmt(s['rejected_all_wrong'])}",
        f"Acceptance rate: {_fmt(rate)}",
        f"Optimizer updates: {_fmt(s['optimizer_updates'])}",
        f"Trained completion tokens (approx.): {_fmt(s['trained_completion_tokens_approx'])}",
        f"Average completion tokens: {_fmt(s['average_completion_tokens'])}",
        f"Truncation rate: {_fmt(s['truncation_rate'])}",
        f"Train answers (malformed = not exactly one label word): {_fmt(s['train_rollouts'])}",
        f"Eval answers: {_fmt(s['eval_rollouts'])}",
        f"Format snapshots in metrics.jsonl: {_fmt(s['format_snapshots'])}",
        f"Initial dev accuracy: {_fmt(s['initial_dev_accuracy'])}",
        f"Best dev accuracy: {_fmt(s['best_dev_accuracy'])}",
        f"Final dev accuracy: {_fmt(s['final_dev_accuracy'])}",
        f"Final checkpoint: {_fmt(s['final_checkpoint'])}",
    ]
    return "\n".join(lines)


def write_summary(log_dir: Path, final_checkpoint: Optional[str] = None) -> dict[str, Any]:
    summary = summarize(log_dir, final_checkpoint)
    (log_dir / SUMMARY_FILE).write_text(summary["text"] + "\n")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize an RL run from its log folder.")
    parser.add_argument("--log-path", required=True, help="the run's log folder")
    parser.add_argument("--final-checkpoint", default=None, help="e.g. step-12")
    args = parser.parse_args(argv)
    summary = write_summary(Path(args.log_path), args.final_checkpoint)
    print(summary["text"])
    print(f"Wrote {Path(args.log_path) / SUMMARY_FILE}")


if __name__ == "__main__":
    main()
