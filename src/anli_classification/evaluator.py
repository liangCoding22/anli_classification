from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from anli_classification.classifier import NliClassifier
from anli_classification.metrics import MetricsCalculator
from anli_classification.schemas import Example, Prediction
from anli_classification.store import PredictionStore


class Evaluator:
    """Runs a classifier over named slices of examples, resuming from a cache."""

    def __init__(
        self,
        classifier: NliClassifier,
        metrics: MetricsCalculator,
        run_dir: Path,
        batch_size: int = 256,
    ) -> None:
        self.classifier = classifier
        self.metrics = metrics
        self.run_dir = run_dir
        self.batch_size = batch_size

    def evaluate(self, slices: dict[str, Sequence[Example]]) -> dict[str, dict[str, Any]]:
        report: dict[str, dict[str, Any]] = {}
        everything: list[Prediction] = []
        for name, examples in slices.items():
            preds = self._run_slice(name, examples)
            report[name] = self.metrics.compute(preds)
            everything.extend(preds)
        report["overall"] = self.metrics.compute(everything)
        self._write_report(report)
        return report

    def _run_slice(self, name: str, examples: Sequence[Example]) -> list[Prediction]:
        store = PredictionStore(self.run_dir / f"{name}.jsonl")
        cached = store.load()
        todo = [ex for ex in examples if ex.key not in cached or not cached[ex.key].ok]
        fresh: dict[str, Prediction] = {}
        for start in range(0, len(todo), self.batch_size):
            preds = self.classifier.classify_batch(todo[start : start + self.batch_size])
            store.append(preds)
            fresh.update((p.key, p) for p in preds)
        final = {**cached, **fresh}
        return [final[ex.key] for ex in examples if ex.key in final]

    def _write_report(self, report: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "metrics.json"
        path.write_text(json.dumps(report, indent=2))
