from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

LABELS: tuple[str, ...] = ("entailment", "neutral", "contradiction")
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)


@dataclass(frozen=True)
class Example:
    split: str
    round_id: str
    uid: str
    premise: str
    hypothesis: str
    label: int

    @property
    def key(self) -> str:
        """Globally unique id: ANLI rounds are separate datasets with their own uids."""
        return f"{self.split}_{self.round_id}/{self.uid}"


@dataclass(frozen=True)
class Prediction:
    key: str
    gold: int
    pred: Optional[int]
    raw: str
    error: Optional[str] = None
    reasoning: str = ""  # model's thinking before the answer, when the API returns it

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prediction":
        return cls(**data)
