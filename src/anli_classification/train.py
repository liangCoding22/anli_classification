from __future__ import annotations

import argparse

from transformers import Trainer, TrainingArguments, set_seed

from anli_classification.config import TrainConfig
from anli_classification.data import load_anli_splits, tokenize_dataset
from anli_classification.metrics import compute_metrics
from anli_classification.model import load_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Fine-tune a classifier on ANLI.")
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=defaults.eval_batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--epochs", type=float, default=defaults.epochs)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    set_seed(config.seed)

    tokenizer = load_tokenizer(config.model_name)
    model = load_model(config.model_name)
    raw = load_anli_splits(config.train_rounds, config.eval_rounds)
    tokenized = tokenize_dataset(raw, tokenizer, config.max_length)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
