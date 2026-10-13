from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

from anli_classification.config import TrainConfig
from anli_classification.data import load_anli_splits, tokenize_dataset
from anli_classification.metrics import compute_metrics
from anli_classification.model import load_tokenizer


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Evaluate a saved ANLI checkpoint.")
    parser.add_argument("--checkpoint", default=defaults.output_dir)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--eval-batch-size", type=int, default=defaults.eval_batch_size)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig()
    tokenizer = load_tokenizer(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    raw = load_anli_splits(config.train_rounds, config.eval_rounds)
    tokenized = tokenize_dataset(raw, tokenizer, args.max_length)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="outputs/eval",
            per_device_eval_batch_size=args.eval_batch_size,
            report_to=[],
        ),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    results = {}
    for round_id in config.eval_rounds:
        split = f"test_{round_id}"
        metrics = trainer.evaluate(tokenized[split], metric_key_prefix=split)
        results[split] = metrics
        print(f"{split}: {metrics}")

    out_path = Path(args.checkpoint) / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
