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
    reason: str = ""  # annotator's explanation of the gold label (ANLI 'reason' column)

    @property
    def key(self) -> str:
        """Globally unique id: ANLI rounds are separate datasets with their own uids."""
        return f"{self.split}_{self.round_id}/{self.uid}"


SAMPLE_SEP = "#"


def sample_key(example_key: str, sample: int) -> str:
    """Key for the `sample`-th answer to an example. Sample 0 keeps the bare key, so single-sample runs stay compatible."""
    return example_key if sample == 0 else f"{example_key}{SAMPLE_SEP}{sample}"


def example_key_of(key: str) -> str:
    """Inverse of `sample_key`: the example key a prediction key belongs to."""
    return key.split(SAMPLE_SEP, 1)[0]


@dataclass(frozen=True)
class Prediction:
    key: str
    gold: int
    pred: Optional[int]
    raw: str
    error: Optional[str] = None
    reasoning: str = ""  # model's thinking before the answer, when the API returns it
    premise: str = ""  # copied from the example so a bad prediction can be read in place
    hypothesis: str = ""
    reason: str = ""  # gold label rationale, to compare against the model's reasoning

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prediction":
        return cls(**data)
