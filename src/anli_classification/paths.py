"""Where the pipeline reads and writes, in one place.

Everything lives under one output root: `outputs/` by default, or `$ANLI_OUTPUT_DIR`. Scripts take their own flags for the
pool, traces and prompt locations; these are only the defaults, so a command without flags finds the same files as before.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL_DIR = "qwen3-4b"
DEFAULT_PROMPT_DIR = "zero-shot-v3"
DEFAULT_POOL_RUN = "train_r3_n50000_seed0_excl"  # the N50000 pool the reported runs used


def output_root() -> Path:
    return Path(os.environ.get("ANLI_OUTPUT_DIR", "outputs"))


def training_pool_root(root: Path | None = None) -> Path:
    """Directory holding one subdirectory per `anli-training-pool` run."""
    return (root or output_root()) / DEFAULT_MODEL_DIR / DEFAULT_PROMPT_DIR / "training_pool"


def default_pool_path(root: Path | None = None) -> Path:
    return training_pool_root(root) / DEFAULT_POOL_RUN / "pool_mixed.jsonl"


def teacher_traces_glob(root: Path | None = None) -> str:
    return str((root or output_root()) / "teacher_traces" / "*" / "*" / "traces.jsonl")


def teacher_prompts_glob(root: Path | None = None) -> str:
    return str((root or output_root()) / "teacher_prompts" / "*.jsonl")
