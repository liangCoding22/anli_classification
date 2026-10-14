"""List and promote checkpoints of an RL trainer job (Fireworks control-plane API).

Each saved checkpoint is one row. A row is promotable when it is a base sampler snapshot;
promotion turns that snapshot into a model you can deploy and evaluate, even after the trainer
job has finished (as long as the retention window has not expired).

    anli-rl-checkpoints list --job-id training-api-service-xxxx
    anli-rl-checkpoints promote --job-id training-api-service-xxxx --checkpoint step-2-xxxx \
        --output-model-id anli-qwen3-4b-step2 --base-model accounts/fireworks/models/qwen3-4b

Promotion is explicit: nothing is promoted automatically.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional, Sequence

from anli_classification.config import DEFAULT_MODEL

DEFAULT_BASE_URL = "https://api.fireworks.ai"
_OUTPUT_MODEL_ID_MAX = 63


def _manager() -> Any:
    from fireworks.training.sdk.trainer import TrainerJobManager

    base_url = os.environ.get("FIREWORKS_BASE_URL", DEFAULT_BASE_URL)
    return TrainerJobManager(api_key=os.environ["FIREWORKS_API_KEY"], base_url=base_url)


def short_name(row: dict[str, Any]) -> str:
    """The checkpoint's own name, e.g. 'step-2-8377bd45' from '.../checkpoints/step-2-8377bd45'."""
    return row["name"].rsplit("/", 1)[-1]


def list_rows(job_id: str, manager: Optional[Any] = None) -> list[dict[str, Any]]:
    return (manager or _manager()).list_checkpoints(job_id)


def find_row(rows: Sequence[dict[str, Any]], checkpoint: str) -> dict[str, Any]:
    matches = [row for row in rows if short_name(row) == checkpoint]
    if not matches:
        known = ", ".join(short_name(row) for row in rows) or "none"
        raise SystemExit(f"checkpoint {checkpoint!r} not found on this trainer job. Known: {known}")
    return matches[0]


def validate_output_model_id(output_model_id: str) -> None:
    """Fireworks model ids: 1-63 characters, lowercase letters, digits, and hyphens."""
    import re

    if len(output_model_id) > _OUTPUT_MODEL_ID_MAX or not re.fullmatch(r"[a-z0-9-]+", output_model_id):
        raise SystemExit(
            f"invalid --output-model-id {output_model_id!r}: 1-{_OUTPUT_MODEL_ID_MAX} chars, "
            "lowercase letters, digits and hyphens only"
        )


def promote(
    job_id: str,
    checkpoint: str,
    output_model_id: str,
    base_model: str,
    manager: Optional[Any] = None,
) -> dict[str, Any]:
    validate_output_model_id(output_model_id)
    manager = manager or _manager()
    row = find_row(manager.list_checkpoints(job_id), checkpoint)
    if not row.get("promotable"):
        raise SystemExit(
            f"checkpoint {checkpoint!r} is not promotable (type {row.get('checkpointType')}). "
            "Only base sampler snapshots can be promoted."
        )
    return manager.promote_checkpoint(
        name=row["name"], output_model_id=output_model_id, base_model=base_model
    )


def format_rows(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "no checkpoints"
    lines = [f"{'checkpoint':<28} {'promotable':<11} {'type':<34} created"]
    for row in rows:
        lines.append(
            f"{short_name(row):<28} {str(bool(row.get('promotable'))):<11} "
            f"{str(row.get('checkpointType', '')):<34} {row.get('createTime', '')}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="List or promote checkpoints of an RL trainer job.")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list checkpoints of a trainer job")
    listing.add_argument("--job-id", required=True, help="short trainer job id, e.g. training-api-service-xxxx")
    promoting = sub.add_parser("promote", help="promote one promotable checkpoint to a model")
    promoting.add_argument("--job-id", required=True)
    promoting.add_argument("--checkpoint", required=True, help="checkpoint name, e.g. step-2-xxxx")
    promoting.add_argument("--output-model-id", required=True)
    promoting.add_argument("--base-model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    if args.command == "list":
        print(format_rows(list_rows(args.job_id)))
    else:
        model = promote(args.job_id, args.checkpoint, args.output_model_id, args.base_model)
        print(f"Promoted {args.checkpoint} -> {args.output_model_id}")
        print(f"state: {model.get('state')}  name: {model.get('name')}")


if __name__ == "__main__":
    main()
