"""Turns training traces into chat-format SFT examples (OpenAI messages, as the cookbook's sft_loop reads).

Each trace becomes three messages, with the prompt built exactly as the eval builds it (zero-shot-v3):

- system: NliPromptV3.SYSTEM
- user: NliPromptV3.USER with the premise and hypothesis filled in
- assistant: content = the label, reasoning_content = the thinking (the text between <think> and </think>)

The cookbook's data path turns reasoning_content into a thinking part, and the Qwen3 renderer then
writes "<think>\\n{reasoning}\\n</think>\\n\\n{label}", the same tokens as a literal <think> string.
The reasoning is stripped so the template's own newlines are not doubled.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from anli_classification.prompts import NliPromptV3
from anli_classification.schemas import LABELS, Example

DEFAULT_TRACES = Path("outputs/qwen3-4b/zero-shot-v3/training_data/train_traces.jsonl")
DEFAULT_OUTPUT = Path("outputs/qwen3-4b/zero-shot-v3/sft_data")
SFT_FILE = "train_sft.jsonl"
SUMMARY_FILE = "summary.json"


def to_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Messages for one trace. The assistant label is the original gold label, not the parsed answer."""
    example = Example(
        split="train",
        round_id="r3",
        uid=record["key"],
        premise=record["premise"],
        hypothesis=record["hypothesis"],
        label=record["gold"],
        reason=record["reason"],
    )
    messages = NliPromptV3().build(example)
    messages.append({
        "role": "assistant",
        "content": LABELS[record["gold"]],
        "reasoning_content": record["trace"]["reasoning"].strip(),
    })
    return messages


def skip_reason(record: dict[str, Any]) -> Optional[str]:
    """Why a trace cannot be used for SFT, or None. Such traces are counted, not written."""
    if not record["trace"]["reasoning"].strip():
        return "empty_reasoning"
    if record["trace"]["pred"] != LABELS[record["gold"]]:
        return "pred_not_gold"
    return None


def build(
    traces_path: Path,
    output_dir: Path,
    drop_all_correct_labels: Sequence[str] = (),
) -> dict[str, Any]:
    records = [json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()]
    dropped = Counter(
        r["label"] for r in records
        if r["category"] == "all_correct" and r["label"] in drop_all_correct_labels
    )
    records = [
        r for r in records
        if not (r["category"] == "all_correct" and r["label"] in drop_all_correct_labels)
    ]
    skipped: Counter = Counter()
    kept = []
    for record in records:
        reason = skip_reason(record)
        if reason:
            skipped[reason] += 1
        else:
            kept.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / SFT_FILE).open("w") as f:
        for record in kept:
            f.write(json.dumps({"messages": to_messages(record)}, ensure_ascii=False) + "\n")
    summary = {
        "traces": str(traces_path),
        "examples": len(kept),
        "labels": dict(Counter(r["label"] for r in kept)),
        "skipped": dict(skipped),
        "dropped_all_correct": dict(dropped),
        "drop_all_correct_labels": list(drop_all_correct_labels),
        "prompt_version": NliPromptV3.VERSION,
    }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chat-format SFT data from training traces.")
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--drop-all-correct-label", nargs="*", default=[], choices=("entailment", "neutral", "contradiction"),
        help="drop all-correct traces with these labels (mixed traces are always kept)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = build(args.traces, args.output_dir, args.drop_all_correct_label)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_dir / SFT_FILE}")


if __name__ == "__main__":
    main()
