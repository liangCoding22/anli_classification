from __future__ import annotations

import json
from pathlib import Path

from anli_classification.schemas import Prediction


class PredictionStore:
    """Append-only JSONL cache of predictions, one file per evaluation slice.

    When a key appears more than once (a retry after an error), the last line wins.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Prediction]:
        latest: dict[str, Prediction] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    pred = Prediction.from_dict(json.loads(line))
                    latest[pred.key] = pred
        return latest

    def upsert(self, predictions: list[Prediction]) -> None:
        """Merge predictions into the file, keeping only the latest line per key."""
        merged = self.load()
        for pred in predictions:
            merged[pred.key] = pred
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(p.to_dict()) + "\n" for p in merged.values()))
        tmp.replace(self.path)

    def append(self, predictions: list[Prediction]) -> None:
        if not predictions:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            for pred in predictions:
                fh.write(json.dumps(pred.to_dict()) + "\n")
