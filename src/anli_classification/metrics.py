from __future__ import annotations

from collections import Counter
from math import comb
from typing import Any, Optional, Sequence

from anli_classification.parsers import is_well_formed
from anli_classification.schemas import LABELS, NUM_LABELS, SAMPLE_SEP, Prediction, example_key_of


def pass_at_k(predictions: Sequence[Prediction], ks: Sequence[int] = (1, 3, 5, 8)) -> dict[str, Any]:
    """Unbiased pass@k over repeated samples per example (Chen et al., 2021).

    For an example with n samples of which c are correct, pass@k = 1 - C(n-c, k) / C(n, k).
    Examples with fewer than k samples are left out of that k.
    """
    hits_by_example: dict[str, list[bool]] = {}
    for p in predictions:
        hits_by_example.setdefault(example_key_of(p.key), []).append(p.ok and p.pred == p.gold)

    result: dict[str, Any] = {"examples": len(hits_by_example)}
    for k in ks:
        scores = [
            1 - comb(len(hits) - sum(hits), k) / comb(len(hits), k)
            for hits in hits_by_example.values()
            if len(hits) >= k
        ]
        result[f"pass@{k}"] = sum(scores) / len(scores) if scores else None
    return result


def maj_at_k(predictions: Sequence[Prediction], ks: Sequence[int] = (1, 3, 5, 8)) -> dict[str, Any]:
    """Majority-vote accuracy over the first k samples of each example (maj@k).

    The label predicted most often among the first k samples (by sample index) is the vote. An
    unparsed or errored sample casts no vote. A tie splits the score evenly among the tied labels,
    so it scores 1/len(tied) when the gold label is among them. Examples with fewer than k samples
    are left out of that k.
    """
    samples_by_example: dict[str, list[tuple[int, Optional[int], int]]] = {}
    for p in predictions:
        key, _, index = p.key.partition(SAMPLE_SEP)
        samples_by_example.setdefault(key, []).append((int(index) if index else 0, p.pred if p.ok else None, p.gold))

    result: dict[str, Any] = {"examples": len(samples_by_example)}
    for k in ks:
        scores = []
        for samples in samples_by_example.values():
            if len(samples) < k:
                continue
            first = sorted(samples)[:k]
            votes = Counter(pred for _, pred, _ in first if pred is not None)
            if not votes:
                scores.append(0.0)
                continue
            top = max(votes.values())
            tied = [label for label, count in votes.items() if count == top]
            scores.append(1 / len(tied) if first[0][2] in tied else 0.0)
        result[f"maj@{k}"] = sum(scores) / len(scores) if scores else None
    return result


def format_metrics_line(name: str, m: dict[str, Any]) -> str:
    """One console line summarising a slice's metrics dict (from MetricsCalculator.compute)."""
    line = (
        f"{name:10s} n={m['n']:5d} acc={m['accuracy']:.4f} strict_acc={m['strict_accuracy']:.4f} "
        f"macro_f1={m['macro_f1']:.4f} malformed={m['malformed']} unparsed={m['unparsed']} errors={m['errors']}"
    )
    for name_, value in {**m.get("pass_at_k", {}), **m.get("maj_at_k", {})}.items():
        if name_.startswith(("pass@", "maj@")) and value is not None:
            line += f" {name_}={value:.4f}"
    return line


def no_reasoning_warning(name: str, m: dict[str, Any]) -> Optional[str]:
    """A warning when no reply in a slice reasoned, else None.

    Qwen3 reasons by default, so zero reasoning means the server's chat template put the model in
    non-thinking mode, as it does for promoted RL checkpoints. Such a score does not compare with
    the RL run's own dev evaluation.
    """
    answered = m["n"] - m["errors"]
    if answered == 0 or m.get("with_reasoning", 0) > 0:
        return None
    return (
        f"WARNING {name}: none of {answered} replies has reasoning; the server template likely "
        "disabled thinking. For an RL checkpoint, re-run with --template client to match training."
    )


def is_malformed(prediction: Prediction) -> bool:
    """True when a reply came back but is not exactly one label word (see parsers.is_well_formed)."""
    return prediction.ok and not is_well_formed(prediction.raw)


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

        # Malformed answers (several labels, label not first, no label, empty) count as wrong here.
        malformed = [p for p in predictions if is_malformed(p)]
        strict_correct = sum(p.pred == p.gold and not is_malformed(p) for p in predictions)
        return {
            "n": n,
            "accuracy": correct / n if n else 0.0,
            "strict_accuracy": strict_correct / n if n else 0.0,
            "malformed": len(malformed),
            "macro_f1": f1_total / NUM_LABELS,
            "unparsed": sum(p.pred is None for p in predictions),
            "errors": sum(not p.ok for p in predictions),
            "with_reasoning": sum(p.ok and bool(p.reasoning.strip()) for p in predictions),
            "per_class": per_class,
            "confusion": {"rows_gold_cols_pred": LABELS, "matrix": matrix},
            "pass_at_k": pass_at_k(predictions),
            "maj_at_k": maj_at_k(predictions),
        }
