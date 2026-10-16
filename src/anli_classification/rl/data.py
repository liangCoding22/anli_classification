"""Training pool rows and the fixed dev subset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anli_classification.prompts import NliPromptV3, PROMPTS
from anli_classification.schemas import Example


def load_pool_rows(pool_path: Path, prompt_version: str = NliPromptV3.VERSION) -> list[dict[str, Any]]:
    """One training row per pool example: chat messages for the prompt, plus the gold label."""
    prompt = PROMPTS[prompt_version]()
    rows = []
    for line in pool_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        example = Example(
            split="train",
            round_id="r3",
            uid=record["key"].split("/", 1)[1],
            premise=record["premise"],
            hypothesis=record["hypothesis"],
            label=record["gold"],
            reason=record.get("reason", ""),
        )
        rows.append({"key": record["key"], "gold": record["gold"], "messages": prompt.build(example),
                     "premise": record["premise"], "hypothesis": record["hypothesis"]})
    return rows


def load_dev_rows(n: int, seed: int, prompt_version: str = NliPromptV3.VERSION) -> list[dict[str, Any]]:
    """Fixed dev subset for intermediate evaluation: `n` random dev_r3 examples, same rows every time."""
    from anli_classification.data import AnliDataset

    prompt = PROMPTS[prompt_version]()
    examples = AnliDataset().load("dev", "r3", limit=n, seed=seed)
    return [{"key": ex.key, "gold": ex.label, "messages": prompt.build(ex)} for ex in examples]
