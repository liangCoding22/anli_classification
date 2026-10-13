from __future__ import annotations

from typing import Any, Sequence

from anli_classification.schemas import LABELS, NUM_LABELS, Prediction


def format_metrics_line(name: str, m: dict[str, Any]) -> str:
    """One console line summarising a slice's metrics dict (from MetricsCalculator.compute)."""
    return (
        f"{name:10s} n={m['n']:5d} acc={m['accuracy']:.4f} "
        f"macro_f1={m['macro_f1']:.4f} unparsed={m['unparsed']} errors={m['errors']}"
    )


class MetricsCalculator:
    """Accuracy, macro-F1, per-class scores and confusion matrix for 3-way NLI.

    A prediction with no parsed label (`pred is None`, including request errors)
    counts as wrong and is left out of the confusion matrix.
    """

    def compute(self, predictions: Sequence[Prediction]) -> dict[str, Any]:
        n = len(predictions)
        correct = sum(p.pred == p.gold for p in predictions)
        per_class = {}
        f1_total = 0.0
        for c in range(NUM_LABELS):
            tp = sum(p.gold == c and p.pred == c for p in predictions)
            n_pred = sum(p.pred == c for p in predictions)
            n_gold = sum(p.gold == c for p in predictions)
            precision = tp / n_pred if n_pred else 0.0
            recall = tp / n_gold if n_gold else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            f1_total += f1
            per_class[LABELS[c]] = {"precision": precision, "recall": recall, "f1": f1}

        matrix = [[0] * NUM_LABELS for _ in range(NUM_LABELS)]
        for p in predictions:
            if p.pred is not None:
                matrix[p.gold][p.pred] += 1

        return {
            "n": n,
            "accuracy": correct / n if n else 0.0,
            "macro_f1": f1_total / NUM_LABELS,
            "unparsed": sum(p.pred is None for p in predictions),
            "errors": sum(not p.ok for p in predictions),
            "per_class": per_class,
            "confusion": {"rows_gold_cols_pred": LABELS, "matrix": matrix},
        }
