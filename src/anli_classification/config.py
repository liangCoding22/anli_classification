from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    output_dir: str = "outputs/anli-distilbert"
    max_length: int = 128
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    epochs: float = 1.0
    weight_decay: float = 0.01
    seed: int = 42
    # Combine ANLI rounds for train/dev; evaluate each test round separately.
    train_rounds: tuple[str, ...] = ("r1", "r2", "r3")
    eval_rounds: tuple[str, ...] = ("r1", "r2", "r3")
