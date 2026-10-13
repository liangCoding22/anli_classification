from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from anli_classification.classifier import NliClassifier
from anli_classification.config import DEFAULT_MODEL, RunConfig
from anli_classification.data import AnliDataset, ROUNDS, SPLITS
from anli_classification.evaluator import Evaluator
from anli_classification.llm.base import ChatBackend
from anli_classification.llm.fireworks import FireworksBatchBackend, FireworksChatBackend
from anli_classification.metrics import MetricsCalculator, format_metrics_line
from anli_classification.parsers import LabelParser
from anli_classification.prompts import NliPrompt


def build_backend(config: RunConfig) -> ChatBackend:
    if config.backend == "batch":
        if not config.account_id:
            raise SystemExit("--account-id is required for the batch backend")
        return FireworksBatchBackend(
            config.model,
            account_id=config.account_id,
            artifacts_dir=config.run_dir / "batch_jobs",
        )
    return FireworksChatBackend(config.model, max_workers=config.concurrency)


def build_classifier(config: RunConfig) -> NliClassifier:
    return NliClassifier(
        prompt=NliPrompt(),
        backend=build_backend(config),
        parser=LabelParser(),
        max_tokens=config.max_tokens,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot ANLI classification.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("chat", "batch"), default="chat")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--rounds", nargs="+", choices=ROUNDS, default=list(ROUNDS))
    parser.add_argument("--limit", type=int, default=None, help="random N per slice")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(argv)
    return RunConfig(
        model=args.model,
        backend=args.backend,
        splits=tuple(args.splits),
        rounds=tuple(args.rounds),
        limit=args.limit,
        seed=args.seed,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        account_id=args.account_id,
        output_dir=Path(args.output_dir),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    dataset = AnliDataset()
    slices = {
        f"{split}_{round_id}": dataset.load(split, round_id, config.limit, config.seed)
        for split in config.splits
        for round_id in config.rounds
    }
    evaluator = Evaluator(
        build_classifier(config), MetricsCalculator(), config.run_dir, config.batch_size
    )
    report = evaluator.evaluate(slices)
    for name, m in report.items():
        print(format_metrics_line(name, m))
    print(f"Wrote {config.run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
