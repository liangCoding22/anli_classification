"""Sample teacher traces for a prompt file with Fireworks batch jobs (resumable, chunked, chunks in parallel).

    PYTHONPATH=src .venv-sft/bin/python scripts/generate_teacher_traces.py \
        --prompts outputs/teacher_prompts/batch1_prompts.jsonl --run-name batch1 --account-id <account>

Each prompt row needs `key`, `gold`, `premise`, `hypothesis`, and `samples` (answers to draw; default
--samples). Requests use the zero-shot-v3 prompt, the same as eval and SFT. Every answer is stored with its
finish reason, token count and parsed label, so correct traces can be picked afterwards:

    outputs/teacher_traces/<model-slug>/<run-name>/chunk_XX/   input.jsonl, output.jsonl, job.json
    outputs/teacher_traces/<model-slug>/<run-name>/traces.jsonl  one line per answer

A rerun skips chunks whose output.jsonl is already there, so an interrupted run can continue.
"""

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anli_classification.llm.base import ChatRequest
from anli_classification.llm.fireworks import FireworksBatchBackend
from anli_classification.paths import output_root
from anli_classification.teacher_data import build_requests, collect



def download_with_retry(backend: FireworksBatchBackend, dataset: str, tries: int = 6) -> str:
    for attempt in range(tries):
        try:
            return backend.download_dataset(dataset)
        except Exception as exc:  # network errors on large downloads
            print(f"download retry {attempt + 1}/{tries}: {type(exc).__name__}", flush=True)
            time.sleep(15)
    raise RuntimeError(f"could not download {dataset}")


def run_chunk(model: str, account_id: str, chunk_dir: Path, requests: list[ChatRequest]) -> None:
    """Submit one chunk as a batch job. A job already created for this chunk is reused, never resubmitted."""
    if (chunk_dir / "output.jsonl").exists():
        print(f"{chunk_dir.name}: output.jsonl exists, skipping", flush=True)
        return
    jobs_dir = chunk_dir.parent / "jobs" / chunk_dir.name
    backend = FireworksBatchBackend(model, account_id=account_id, artifacts_dir=jobs_dir, poll_seconds=60)
    finished = [d for d in sorted(jobs_dir.glob("anli-job-*")) if (d / "job.json").exists()] if jobs_dir.exists() else []
    if finished:  # an earlier run created and finished this chunk's job; only the download failed
        job = json.loads((finished[-1] / "job.json").read_text())
        print(f"{chunk_dir.name}: reusing finished job {finished[-1].name}", flush=True)
    else:
        print(f"{chunk_dir.name}: submitting {len(requests)} requests", flush=True)
        try:
            backend.complete(requests)
        except Exception as exc:
            # complete() saves job.json before it downloads, so a download error leaves a finished job to reuse.
            finished = [d for d in sorted(jobs_dir.glob("anli-job-*")) if (d / "job.json").exists()]
            if not finished:
                raise
            print(f"{chunk_dir.name}: {type(exc).__name__} after the job finished; downloading again", flush=True)
        finished = [d for d in sorted(jobs_dir.glob("anli-job-*")) if (d / "job.json").exists()]
        job = json.loads((finished[-1] / "job.json").read_text())
    if not (finished[-1] / "output.jsonl").exists():
        (finished[-1] / "output.jsonl").write_text(download_with_retry(backend, job["outputDatasetId"]))
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "output.jsonl").write_text((finished[-1] / "output.jsonl").read_text())
    print(f"{chunk_dir.name}: done ({finished[-1].name})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="accounts/fireworks/models/qwen3p5-397b-a17b")
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--samples", type=int, default=2, help="answers per prompt unless the row has `samples`")
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--chunks", type=int, default=4, help="batch jobs to split the requests into (run in parallel)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", type=Path, default=output_root() / "teacher_traces")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    requests = build_requests(rows, args.samples, args.max_tokens, args.temperature, args.top_p)
    random.Random(args.seed).shuffle(requests)  # every chunk gets the same mix of prompts
    size = -(-len(requests) // args.chunks)
    parts = [requests[i : i + size] for i in range(0, len(requests), size)]
    run_dir = args.out_root / args.model.split("/")[-1] / args.run_name
    print(f"{len(rows)} prompts -> {len(requests)} requests in {len(parts)} chunks of up to {size}; out: {run_dir}", flush=True)
    if args.dry_run:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(parts)) as pool:
        futures = [pool.submit(run_chunk, args.model, args.account_id, run_dir / f"chunk_{i:02d}", part) for i, part in enumerate(parts)]
        for f in futures:
            f.result()
    written = collect(run_dir, {r["key"]: r["gold"] for r in rows})
    print(f"wrote {written} answers to {run_dir / 'traces.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
