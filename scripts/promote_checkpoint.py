"""Promote one promotable checkpoint of a trainer job to a Fireworks model.

    PYTHONPATH=src .venv-sft/bin/python scripts/promote_checkpoint.py \
        --job-id training-api-service-87f517a1 \
        --checkpoint step-141-471a1664 \
        --output-model-id anli-sft-step141

Run with `--list` first to see checkpoint names and which ones are promotable.
Promotion is a one-time, outward-facing action: it creates a model in your account.
"""

import argparse

from anli_classification.rl_checkpoints import DEFAULT_MODEL, format_rows, list_rows, promote


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a trainer checkpoint to a model.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--list", action="store_true", help="list checkpoints and exit")
    parser.add_argument("--checkpoint", help="full checkpoint name, e.g. step-141-471a1664")
    parser.add_argument("--output-model-id", help="name for the new model")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.list:
        print(format_rows(list_rows(args.job_id)))
        return
    if not (args.checkpoint and args.output_model_id):
        parser.error("--checkpoint and --output-model-id are required unless --list is given")

    model = promote(args.job_id, args.checkpoint, args.output_model_id, args.base_model)
    print(f"promoted {args.checkpoint} -> {args.output_model_id}")
    print(f"state: {model.get('state')}  name: {model.get('name')}")


if __name__ == "__main__":
    main()
