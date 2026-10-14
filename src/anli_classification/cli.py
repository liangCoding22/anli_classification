from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from anli_classification.classifier import NliClassifier
from anli_classification.config import DEFAULT_MODEL, TEMPLATES, RunConfig
from anli_classification.data import AnliDataset, DEFAULT_SPLITS, ROUNDS, SPLITS
from anli_classification.evaluator import Evaluator
from anli_classification.llm.base import ChatBackend
from anli_classification.llm.fireworks import FireworksBatchBackend, FireworksChatBackend
from anli_classification.metrics import MetricsCalculator, format_metrics_line, no_reasoning_warning
from anli_classification.parsers import LabelParser
from anli_classification.prompts import PROMPTS, NliPromptV3
from anli_classification.rendering import DEFAULT_TOKENIZER_MODEL, ChatTemplateRenderer


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
        prompt=PROMPTS[config.prompt_version](),
        backend=build_backend(config),
        parser=LabelParser(),
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        renderer=ChatTemplateRenderer(config.tokenizer_model) if config.template == "client" else None,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot ANLI classification.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", choices=tuple(PROMPTS), default=NliPromptV3.VERSION)
    parser.add_argument("--backend", choices=("chat", "batch"), default="chat")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(DEFAULT_SPLITS))
    parser.add_argument("--rounds", nargs="+", choices=ROUNDS, default=list(ROUNDS))
    parser.add_argument("--limit", type=int, default=None, help="random N per slice")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=1, help="answers per example; 3 gives pass@3")
    parser.add_argument("--temperature", type=float, default=0.6, help="must be > 0 for --samples > 1")
    parser.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling cutoff")
    parser.add_argument("--top-k", type=int, default=None, help="top-k cutoff; 0 disables it")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--template", choices=TEMPLATES, default="server",
        help="'client': apply the HF chat template locally with thinking on, as RL training did, "
             "and call /completions (chat backend only). For promoted RL checkpoints, whose "
             "server template disables thinking",
    )
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL,
                        help="HF tokenizer whose chat template --template client applies")
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.template == "client" and args.backend == "batch":
        parser.error("--template client needs --backend chat against a deployment: the batch API "
                     "takes only chat messages, so the server's template always applies")
    return RunConfig(
        model=args.model,
        prompt_version=args.prompt_version,
        backend=args.backend,
        splits=tuple(args.splits),
        rounds=tuple(args.rounds),
        limit=args.limit,
        seed=args.seed,
        max_tokens=args.max_tokens,
        samples=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        account_id=args.account_id,
        output_dir=Path(args.output_dir),
        template=args.template,
        tokenizer_model=args.tokenizer_model,
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
        build_classifier(config),
        MetricsCalculator(),
        config.run_dir,
        config.batch_size,
        config.samples,
    )
    report = evaluator.evaluate(slices)
    for name, m in report.items():
        print(format_metrics_line(name, m))
    for name, m in report.items():
        warning = no_reasoning_warning(name, m) if name != "overall" else None
        if warning:
            print(warning)
    print(f"Wrote {config.run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
