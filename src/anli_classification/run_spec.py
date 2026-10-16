"""What a results folder was produced with, so cached predictions are never reused for different settings.

The evaluator caches predictions per example, and a folder is named only by model, prompt version and template. Without a
record of the sampling settings, a run at temperature 0.0 after one at 0.6 would find every example cached, make no
model calls, and report the old answers as the new run. Each folder therefore gets a `run_spec.json`:

- a new folder (no cached predictions) gets the spec written;
- a folder whose spec matches is reused;
- a folder whose spec differs raises `RunSpecMismatch`; start separate results with `--run-tag NAME`;
- an older folder with predictions but no spec (made before this existed) is reused as before, with a warning,
  because its settings are unknown.

Settings that do not change what a model answers (backend, concurrency, batch size, the deployment serving it, the
data sample) are recorded or ignored, not compared.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

from anli_classification.config import RunConfig, split_model

SPEC_FILE = "run_spec.json"
COMPARED_FIELDS = (
    "model", "prompt_version", "template", "tokenizer_model", "temperature", "top_p", "top_k", "max_tokens", "samples",
)


class RunSpecMismatch(Exception):
    """The folder's cached predictions were made with different settings."""

    def __init__(self, run_dir: Path, differences: dict[str, tuple[Any, Any]]) -> None:
        self.run_dir = run_dir
        self.differences = differences
        lines = [f"  {name}: {old!r} in the cached results, {new!r} now" for name, (old, new) in differences.items()]
        super().__init__(
            f"Cached results in {run_dir} were made with different settings:\n" + "\n".join(lines)
            + "\nReusing them would score the old answers as this run. Pass --run-tag NAME to write this run to a "
              "separate folder."
        )


@dataclass(frozen=True)
class RunSpec:
    model: str  # the model itself, without a '#deployment' part
    prompt_version: str
    template: str
    tokenizer_model: Optional[str]  # only meaningful for template "client"
    temperature: float
    top_p: float
    top_k: Optional[int]
    max_tokens: int
    samples: int
    backend: str = "chat"  # recorded, not compared: batch results are merged into the same folder on purpose
    deployment: Optional[str] = None  # recorded, not compared

    @classmethod
    def from_config(cls, config: RunConfig) -> "RunSpec":
        model, deployment = split_model(config.model)
        return cls(
            model=model,
            prompt_version=config.prompt_version,
            template=config.template,
            tokenizer_model=config.tokenizer_model if config.template == "client" else None,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            samples=config.samples,
            backend=config.backend,
            deployment=deployment,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunSpec":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def differences(self, cached: "RunSpec") -> dict[str, tuple[Any, Any]]:
        """Compared settings that differ, as {name: (cached value, this run's value)}."""
        return {
            name: (getattr(cached, name), getattr(self, name))
            for name in COMPARED_FIELDS
            if getattr(cached, name) != getattr(self, name)
        }


def has_cached_predictions(run_dir: Path) -> bool:
    return run_dir.exists() and any(p.stat().st_size > 0 for p in run_dir.glob("*_r[123].jsonl"))


def ensure_run_spec(run_dir: Path, spec: RunSpec) -> str:
    """Check or write the folder's spec. Returns "new", "match" or "legacy"; raises RunSpecMismatch."""
    path = run_dir / SPEC_FILE
    if path.exists():
        cached = RunSpec.from_dict(json.loads(path.read_text()))
        differences = spec.differences(cached)
        if differences:
            raise RunSpecMismatch(run_dir, differences)
        return "match"
    if has_cached_predictions(run_dir):
        return "legacy"
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2) + "\n")
    return "new"
