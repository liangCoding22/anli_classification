"""Dev accuracy, per-class recall, predicted share and macro-F1."""

from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Sequence

from anli_classification.schemas import LABELS


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
