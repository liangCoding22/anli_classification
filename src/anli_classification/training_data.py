"""Collects training traces from the train_r3 sample runs (N200, N2000, N10000, N50000).

Each example has its sampled answers (sample indices 0..n-1, with thinking in `reasoning`).

- mixed: some answers correct, some wrong (1-3 of 4, or 1-7 of 8). Every mixed example is kept.
- all correct: every answer correct. A random `all_correct_rate` share of these examples is kept,
  drawn across all runs together.
- all wrong, or incomplete (a missing or errored answer): dropped.

Each kept example contributes one trace, chosen at random among its correct answers.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from anli_classification.schemas import LABELS, SAMPLE_SEP, Prediction

RUNS = (
    "train_r3_n200_seed0",
    "train_r3_n2000_seed0_excl",
    "train_r3_n10000_seed0_excl",
    "train_r3_n50000_seed0_excl",
)
ALL_CORRECT_RATE = 0.05
DEFAULT_BASE = Path("outputs/qwen3-4b/zero-shot-v3/training_pool")
DEFAULT_OUTPUT = Path("outputs/qwen3-4b/zero-shot-v3/training_data")
TRACES_FILE = "train_traces.jsonl"
SUMMARY_FILE = "summary.json"


def read_latest(path: Path) -> dict[str, Prediction]:
    """Latest prediction per key, read line by line (the 50k file is large)."""
    latest: dict[str, Prediction] = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                pred = Prediction.from_dict(json.loads(line))
                latest[pred.key] = pred
    return latest


def group_by_example(predictions: dict[str, Prediction]) -> dict[str, dict[int, Prediction]]:
    """example key -> {sample index: prediction}."""
    grouped: dict[str, dict[int, Prediction]] = {}
    for pred in predictions.values():
        key, sep, index = pred.key.partition(SAMPLE_SEP)
        grouped.setdefault(key, {})[int(index) if sep else 0] = pred
    return grouped


def categorize(samples: dict[int, Prediction]) -> str:
    """'all_correct', 'mixed', 'all_wrong', or 'incomplete' (a missing or errored answer)."""
    n = len(samples)
    if sorted(samples) != list(range(n)) or any(not p.ok for p in samples.values()):
        return "incomplete"
    correct = sum(p.pred == p.gold for p in samples.values())
    if correct == n:
        return "all_correct"
    if correct == 0:
        return "all_wrong"
    return "mixed"


def make_record(key: str, source: str, samples: dict[int, Prediction], category: str, index: int) -> dict[str, Any]:
    chosen = samples[index]
    return {
        "key": key,
        "source_run": source,
        "category": category,
        "label": LABELS[chosen.gold],
        "gold": chosen.gold,
        "n_correct": sum(p.pred == p.gold for p in samples.values()),
        "n_samples": len(samples),
        "premise": chosen.premise,
        "hypothesis": chosen.hypothesis,
        "reason": chosen.reason,
        "trace": {
            "sample": index,
            "pred": LABELS[chosen.pred],
            "raw": chosen.raw,
            "reasoning": chosen.reasoning,
        },
    }


def select_traces(
    examples: dict[str, dict[int, Prediction]],
    sources: dict[str, str],
    all_correct_rate: float,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter]:
    """Trace records for the kept examples, plus counts of each category seen."""
    rng = random.Random(seed)
    counts: Counter = Counter()
    records: list[dict[str, Any]] = []
    all_correct: list[str] = []
    for key in sorted(examples):
        samples = examples[key]
        category = categorize(samples)
        counts[category] += 1
        if category == "mixed":
            correct = sorted(i for i, p in samples.items() if p.pred == p.gold)
            records.append(make_record(key, sources[key], samples, category, rng.choice(correct)))
        elif category == "all_correct":
            all_correct.append(key)

    keep = round(all_correct_rate * len(all_correct))
    for key in rng.sample(all_correct, keep):
        samples = examples[key]
        index = rng.choice(sorted(samples))
        records.append(make_record(key, sources[key], samples, "all_correct", index))
    counts["all_correct_kept"] = keep
    return records, counts


def build(
    base: Path,
    runs: Sequence[str],
    output_dir: Path,
    all_correct_rate: float,
    seed: int,
) -> dict[str, Any]:
    examples: dict[str, dict[int, Prediction]] = {}
    sources: dict[str, str] = {}
    per_run: dict[str, dict[str, int]] = {}
    for run in runs:
        grouped = group_by_example(read_latest(base / run / "predictions.jsonl"))
        per_run[run] = dict(Counter(categorize(s) for s in grouped.values()))
        for key, samples in grouped.items():
            if key in examples:
                raise ValueError(f"example {key} appears in more than one run ({sources[key]}, {run})")
            examples[key] = samples
            sources[key] = run

    records, counts = select_traces(examples, sources, all_correct_rate, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / TRACES_FILE).write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = {
        "runs": list(runs),
        "all_correct_rate": all_correct_rate,
        "seed": seed,
        "examples": len(examples),
        "per_run_categories": per_run,
        "categories": dict(counts),
        "kept": len(records),
        "kept_by_category": dict(Counter(r["category"] for r in records)),
        "kept_label_distribution": dict(Counter(r["label"] for r in records)),
    }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect training traces from the train_r3 sample runs.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="folder holding the run folders")
    parser.add_argument("--runs", nargs="+", default=list(RUNS), help="run folder names under --base")
    parser.add_argument("--all-correct-rate", type=float, default=ALL_CORRECT_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not 0.0 <= args.all_correct_rate <= 1.0:
        parser.error("--all-correct-rate must be between 0 and 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = build(args.base, args.runs, args.output_dir, args.all_correct_rate, args.seed)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_dir / TRACES_FILE} and {args.output_dir / SUMMARY_FILE}")


if __name__ == "__main__":
    main()
