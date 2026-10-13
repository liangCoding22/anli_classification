from __future__ import annotations

from typing import Iterable

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase

from anli_classification import LABEL2ID


def load_anli_splits(
    train_rounds: Iterable[str],
    eval_rounds: Iterable[str],
) -> DatasetDict:
    """Load facebook/anli and merge the requested rounds.

    ANLI ships three adversarial rounds (r1–r3). Training concatenates the
    selected train splits; validation concatenates the matching `dev_*` splits.
    Test splits stay separate so each round can be scored on its own.
    """
    raw = load_dataset("anli")

    train = concatenate_datasets(
        [raw[f"train_{round_id}"] for round_id in train_rounds]
    )
    validation = concatenate_datasets(
        [raw[f"dev_{round_id}"] for round_id in train_rounds]
    )

    splits: dict[str, Dataset] = {
        "train": train,
        "validation": validation,
    }
    for round_id in eval_rounds:
        splits[f"test_{round_id}"] = raw[f"test_{round_id}"]
    return DatasetDict(splits)


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> DatasetDict:
    def tokenize_batch(batch: dict) -> dict:
        encoded = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["labels"] = [
            example if isinstance(example, int) else LABEL2ID[example]
            for example in batch["label"]
        ]
        return encoded

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing premise + hypothesis",
    )
    tokenized.set_format("torch")
    return tokenized
