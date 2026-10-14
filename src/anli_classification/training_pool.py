"""Builds a post-training pool from train examples, selected by their pass rate.

Pipeline, for N random train_r3 examples:

1. Round 1: answer each example 4 times.
2. Select every example with 1-3 of 4 correct, plus `edge_rate * N` random examples
   with 4 of 4 correct and as many with 0 of 4 correct.
3. Round 2: answer each selected example 4 more times (samples 4-7).
4. Pool: the 8 answers per selected example, with the distribution of correct counts (0-8).

Answers are stored in one resumable JSONL file, so a rerun only sends missing or failed
requests. Examples with a missing answer after `MAX_ATTEMPTS` tries are left out and counted.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from anli_classification.classifier import NliClassifier
from anli_classification.cli import build_classifier
from anli_classification.config import DEFAULT_MODEL, RunConfig
from anli_classification.data import AnliDataset
from anli_classification.prompts import PROMPTS, NliPromptV3
from anli_classification.schemas import Example, Prediction, example_key_of, sample_key
from anli_classification.store import PredictionStore

ROUND_1 = range(0, 4)  # sample indices answered in the first round
ROUND_2 = range(4, 8)  # sample indices answered in the second round
EDGE_RATE = 0.02  # share of N drawn from the all-correct and all-wrong groups
MAX_ATTEMPTS = 3  # tries per sample before it is treated as missing
GROUP_MIXED = "1-3/4"
GROUP_ALL_CORRECT = "4/4"
GROUP_ALL_WRONG = "0/4"
PREDICTIONS_FILE = "predictions.jsonl"
POOL_FILE = "pool.jsonl"
SUMMARY_FILE = "summary.json"


@dataclass
class Selection:
    group_of: dict[str, str]  # example key -> group, for every selected example
    incomplete: list[str]  # example keys with a missing round-1 answer
    quota: int  # requested examples per edge group
    available: dict[str, int]  # group -> examples that fell into it


def count_correct(preds: Sequence[Optional[Prediction]]) -> Optional[int]:
    """Correct answers among `preds`, or None if any answer is missing or errored."""
    if any(p is None or not p.ok for p in preds):
        return None
    return sum(p.pred == p.gold for p in preds)


def run_samples(
    classifier: NliClassifier,
    store: PredictionStore,
    examples: Sequence[Example],
    indices: Sequence[int],
    batch_size: int,
) -> dict[str, Prediction]:
    """Ensure each example has an ok answer for each sample index; return the latest answer per key."""
    cached = store.load()
    items = [(sample_key(ex.key, s), ex) for ex in examples for s in indices]
    for _ in range(MAX_ATTEMPTS):
        todo = [(key, ex) for key, ex in items if key not in cached or not cached[key].ok]
        if not todo:
            break
        # One request set per call: a batch backend submits one job and waits for it to finish
        # before returning, so the next round cannot start until this round's job is COMPLETED.
        size = batch_size or len(todo)
        for start in range(0, len(todo), size):
            preds = classifier.classify_batch(todo[start : start + size])
            store.append(preds)
            cached.update((p.key, p) for p in preds)
    return cached


def select_examples(
    examples: Sequence[Example],
    round1: dict[str, Prediction],
    edge_rate: float,
    seed: int,
) -> Selection:
    by_group: dict[str, list[Example]] = {GROUP_MIXED: [], GROUP_ALL_CORRECT: [], GROUP_ALL_WRONG: []}
    incomplete: list[str] = []
    for ex in examples:
        preds = [round1.get(sample_key(ex.key, s)) for s in ROUND_1]
        correct = count_correct(preds)
        if correct is None:
            incomplete.append(ex.key)
        elif correct == len(ROUND_1):
            by_group[GROUP_ALL_CORRECT].append(ex)
        elif correct == 0:
            by_group[GROUP_ALL_WRONG].append(ex)
        else:
            by_group[GROUP_MIXED].append(ex)

    quota = max(1, round(edge_rate * len(examples)))
    rng = random.Random(seed)
    group_of = {ex.key: GROUP_MIXED for ex in by_group[GROUP_MIXED]}
    for group in (GROUP_ALL_CORRECT, GROUP_ALL_WRONG):
        pool = by_group[group]
        for ex in rng.sample(pool, min(quota, len(pool))):
            group_of[ex.key] = group
    available = {group: len(members) for group, members in by_group.items()}
    return Selection(group_of=group_of, incomplete=incomplete, quota=quota, available=available)


def distribution(records: Sequence[dict[str, Any]], total: int) -> dict[int, int]:
    counts = Counter(r["n_correct"] for r in records)
    return {k: counts.get(k, 0) for k in range(total + 1)}


def pool_keeps(correct: int, samples: int) -> bool:
    """Keep examples with a mix of right and wrong answers: not all wrong, not all right."""
    return 0 < correct < samples


def make_record(
    example: Example, group: str, answers: Sequence[Optional[Prediction]], indices: Sequence[int], correct: int
) -> dict[str, Any]:
    """One pool record: the example, its group, and each answer with its sample index and round."""
    return {
        "key": example.key,
        "group": group,
        "gold": example.label,
        "premise": example.premise,
        "hypothesis": example.hypothesis,
        "reason": example.reason,
        "n_correct": correct,
        "n_samples": len(answers),
        "samples": [
            {
                "sample": index,
                "round": 1 if index in ROUND_1 else 2,
                "pred": a.pred,
                "correct": a.pred == a.gold,
                "raw": a.raw,
                "reasoning": a.reasoning,
            }
            for index, a in zip(indices, answers)
        ],
    }


def build_training_pool(
    classifier: NliClassifier,
    examples: Sequence[Example],
    out_dir: Path,
    batch_size: int,
    edge_rate: float,
    seed: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Round 1 for every example. Mixed (1-3 of 4) examples go straight to the pool.

    Only the all-correct and all-wrong examples get a second round, and only the sampled
    `edge_rate * N` of each. An edge example enters the pool if round 2 shows it is mixed.
    """
    store = PredictionStore(out_dir / PREDICTIONS_FILE)
    round1 = run_samples(classifier, store, examples, ROUND_1, batch_size)
    round1_correct = [
        count_correct([round1.get(sample_key(ex.key, s)) for s in ROUND_1]) for ex in examples
    ]
    selection = select_examples(examples, round1, edge_rate, seed)

    mixed = [ex for ex in examples if selection.group_of.get(ex.key) == GROUP_MIXED]
    edge = [
        ex for ex in examples
        if selection.group_of.get(ex.key) in (GROUP_ALL_CORRECT, GROUP_ALL_WRONG)
    ]
    total = len(ROUND_1) + len(ROUND_2)
    final = run_samples(classifier, store, edge, range(total), batch_size)

    records: list[dict[str, Any]] = []
    incomplete_final: list[str] = []
    for ex in mixed:
        answers = [round1[sample_key(ex.key, s)] for s in ROUND_1]
        records.append(make_record(ex, GROUP_MIXED, answers, list(ROUND_1), count_correct(answers)))
    for ex in edge:
        answers = [final.get(sample_key(ex.key, s)) for s in range(total)]
        correct = count_correct(answers)
        if correct is None:
            incomplete_final.append(ex.key)
            continue
        records.append(make_record(ex, selection.group_of[ex.key], answers, list(range(total)), correct))

    pool = [r for r in records if pool_keeps(r["n_correct"], r["n_samples"])]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / POOL_FILE).write_text("".join(json.dumps(r) + "\n" for r in pool))

    def by_group(recs: Sequence[dict[str, Any]], group: str, samples: int) -> dict[int, int]:
        return distribution([r for r in recs if r["group"] == group], samples)

    summary = {
        "settings": settings,
        "n_examples": len(examples),
        "round1_distribution": distribution(
            [{"n_correct": c} for c in round1_correct if c is not None], len(ROUND_1)
        ),
        "round1_incomplete": selection.incomplete,
        "selection": {
            "edge_quota": selection.quota,
            "available": selection.available,
            "selected": dict(Counter(selection.group_of.values())),
        },
        "incomplete_final": incomplete_final,
        "pool_size": len(pool),
        "pool_from_mixed": sum(1 for r in pool if r["group"] == GROUP_MIXED),
        "pool_from_edge": sum(1 for r in pool if r["group"] != GROUP_MIXED),
        "excluded_from_pool": len(records) - len(pool),
        "distribution": {
            GROUP_MIXED: by_group(records, GROUP_MIXED, len(ROUND_1)),
            GROUP_ALL_CORRECT: by_group(records, GROUP_ALL_CORRECT, total),
            GROUP_ALL_WRONG: by_group(records, GROUP_ALL_WRONG, total),
        },
        "pool_distribution": {
            GROUP_MIXED: by_group(pool, GROUP_MIXED, len(ROUND_1)),
            GROUP_ALL_CORRECT: by_group(pool, GROUP_ALL_CORRECT, total),
            GROUP_ALL_WRONG: by_group(pool, GROUP_ALL_WRONG, total),
        },
    }
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    selection = summary["selection"]
    lines = [
        f"examples: {summary['n_examples']}  round-1 incomplete: {len(summary['round1_incomplete'])}",
        f"edge quota per group: {selection['edge_quota']}  available: {selection['available']}",
        f"incomplete after round 2: {len(summary['incomplete_final'])}",
        f"final pool: {summary['pool_size']}  (mixed {summary['pool_from_mixed']}, "
        f"edge survivors {summary['pool_from_edge']})  excluded: {summary['excluded_from_pool']}",
        "",
        f"{'mixed 1-3/4 (4 samples)':<26}{'n':>6}{'pool':>8}",
    ]
    for group, samples in ((GROUP_MIXED, len(ROUND_1)), (GROUP_ALL_CORRECT, 8), (GROUP_ALL_WRONG, 8)):
        if group == GROUP_ALL_CORRECT:
            lines += ["", f"{'edge 4/4 and 0/4 (8 samples)':<26}"]
        for k in range(samples + 1):
            dist = summary["distribution"][group]
            pool = summary["pool_distribution"][group]
            if group == GROUP_MIXED:
                lines.append(f"  {group} correct {k}/4{'':<12}{dist[k]:>6}{pool[k]:>8}")
            elif dist[k] or pool[k]:
                lines.append(f"  {group} correct {k}/8{'':<12}{dist[k]:>6}{pool[k]:>8}")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> tuple[RunConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Build a post-training pool: 8 answers per selected train example."
    )
    parser.add_argument("--n", type=int, default=200, help="random train_r3 examples to start from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-rate", type=float, default=EDGE_RATE,
                        help="share of N taken from the 4/4 and 0/4 groups")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", choices=tuple(PROMPTS), default=NliPromptV3.VERSION)
    parser.add_argument("--backend", choices=("chat", "batch"), default="batch")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="requests per batch job; 0 submits each round as one job")
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--exclude-run", nargs="*", default=[],
                        help="training_pool folders whose examples must not be drawn again")
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be at least 1")
    if not 0.0 <= args.edge_rate <= 1.0:
        parser.error("--edge-rate must be between 0 and 1")
    config = RunConfig(
        model=args.model,
        prompt_version=args.prompt_version,
        backend=args.backend,
        seed=args.seed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        account_id=args.account_id,
        output_dir=Path(args.output_dir),
    )
    return config, args


def draw_examples(
    candidates: Sequence[Example], n: int, seed: int, excluded: set[str]
) -> list[Example]:
    """Random sample of `n` candidates whose example key is not in `excluded`, in dataset order."""
    available = [ex for ex in candidates if ex.key not in excluded]
    if n > len(available):
        raise ValueError(f"only {len(available)} examples available, asked for {n}")
    picked = sorted(random.Random(seed).sample(range(len(available)), n))
    return [available[i] for i in picked]


def excluded_keys(run_dir: Path) -> set[str]:
    """Example keys already answered in an earlier run, read from its predictions file."""
    return {example_key_of(k) for k in PredictionStore(run_dir / PREDICTIONS_FILE).load()}


def main(argv: Optional[Sequence[str]] = None) -> None:
    config, args = parse_args(argv)
    excluded = set().union(*(excluded_keys(Path(folder)) for folder in args.exclude_run))
    candidates = AnliDataset().load("train", "r3")
    examples = draw_examples(candidates, args.n, args.seed, excluded)
    name = f"train_r3_n{args.n}_seed{args.seed}" + ("_excl" if excluded else "")
    out_dir = config.run_dir / "training_pool" / name
    settings = {
        "model": config.model,
        "prompt_version": config.prompt_version,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "n": args.n,
        "seed": args.seed,
        "edge_rate": args.edge_rate,
        "excluded_examples": len(excluded),
        "strategy": "mixed (1-3 of 4) go straight to pool; only all-correct/all-wrong get round 2",
        "samples_per_round": len(ROUND_1),
    }
    summary = build_training_pool(
        build_classifier(config), examples, out_dir, config.batch_size, args.edge_rate, args.seed, settings
    )
    print(format_summary(summary))
    print(f"Wrote {out_dir / POOL_FILE} and {out_dir / SUMMARY_FILE}")


if __name__ == "__main__":
    main()
