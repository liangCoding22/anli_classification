from __future__ import annotations

import random
from typing import Optional

from datasets import load_dataset

from anli_classification.schemas import Example

HF_DATASET = "facebook/anli"
ROUNDS = ("r1", "r2", "r3")
SPLITS = ("train", "dev", "test")
DEFAULT_SPLITS = ("dev", "test")  # train is opt-in: it is large and not the evaluation set


class AnliDataset:
    """Loads ANLI splits from the Hugging Face Hub as `Example` objects."""

    def __init__(self, hf_name: str = HF_DATASET) -> None:
        self.hf_name = hf_name

    def load(
        self,
        split: str,
        round_id: str,
        limit: Optional[int] = None,
        seed: int = 0,
    ) -> list[Example]:
        if split not in SPLITS or round_id not in ROUNDS:
            raise ValueError(f"unknown split/round: {split!r}/{round_id!r}")
        rows = load_dataset(self.hf_name, split=f"{split}_{round_id}")
        examples = [
            Example(
                split=split,
                round_id=round_id,
                uid=row["uid"],
                premise=row["premise"],
                hypothesis=row["hypothesis"],
                label=row["label"],
                reason=row["reason"],
            )
            for row in rows
        ]
        if limit is not None and limit < len(examples):
            # Splits are sorted by label; a prefix would be single-class.
            picked = sorted(random.Random(seed).sample(range(len(examples)), limit))
            examples = [examples[i] for i in picked]
        return examples
