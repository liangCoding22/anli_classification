from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Optional

from anli_classification.prompts import PROMPTS, NliPromptV3

DEFAULT_MODEL = "accounts/fireworks/models/qwen3-4b"


def model_slug(model: str) -> str:
    """Directory-safe name for a model, e.g. 'qwen3-4b' for 'accounts/fireworks/models/qwen3-4b'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.split("/")[-1])


def run_dir_for(output_dir: Path, model: str, prompt_version: str = NliPromptV3.VERSION) -> Path:
    """Where a model's predictions and metrics for one prompt version live."""
    return output_dir / model_slug(model) / prompt_version


@dataclass(frozen=True)
class RunConfig:
    model: str = DEFAULT_MODEL
    prompt_version: str = NliPromptV3.VERSION  # key in prompts.PROMPTS
    backend: str = "chat"  # "chat": concurrent requests; "batch": Fireworks batch jobs
    splits: tuple[str, ...] = ("dev", "test")  # keep in sync with data.DEFAULT_SPLITS
    rounds: tuple[str, ...] = ("r1", "r2", "r3")
    limit: Optional[int] = None  # random sample per slice, for smoke tests
    seed: int = 0
    max_tokens: int = 4096  # reasoning models spend tokens before the label
    samples: int = 1  # answers per example; 3 gives pass@3
    temperature: float = 0.6  # >0 is needed for samples to differ
    top_p: float = 0.95  # nucleus sampling cutoff; 1.0 disables it
    concurrency: int = 8  # chat backend only
    batch_size: int = 256  # examples per chunk; one batch job per chunk
    account_id: Optional[str] = None  # batch backend only
    output_dir: Path = field(default=Path("outputs"))

    @property
    def run_dir(self) -> Path:
        return run_dir_for(self.output_dir, self.model, self.prompt_version)
