"""Post-process a finished Fireworks batch job into the same outputs as `anli-eval`.

For a job ID, this fetches (if not already saved) the job record, input rows and
output rows, scores each slice, and writes:

  <run_dir>/<slice>.jsonl                    predictions, merged with other runs
  <run_dir>/batch_jobs/<job_id>/job.json     Fireworks job record
  <run_dir>/batch_jobs/<job_id>/input.jsonl  requests sent (if the API allows downloading it)
  <run_dir>/batch_jobs/<job_id>/output.jsonl raw response rows
  <run_dir>/batch_jobs/<job_id>/metrics.json same shape as a run's metrics.json
  <run_dir>/batch_jobs/<job_id>/summary.json batch metadata and headline metrics
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import requests

from anli_classification.classifier import prediction_from_result
from anli_classification.config import run_dir_for
from anli_classification.data import AnliDataset
from anli_classification.llm.base import ChatResult
from anli_classification.llm.fireworks import FireworksBatchBackend, job_state, parse_batch_results
from anli_classification.metrics import MetricsCalculator, format_metrics_line
from anli_classification.parsers import LabelParser
from anli_classification.prompts import PROMPTS, NliPromptV3
from anli_classification.schemas import Example, Prediction, example_key_of
from anli_classification.store import PredictionStore


def _slice_of(key: str) -> str:
    """Example keys look like 'dev_r1/<uid>'; the slice name is the part before '/'."""
    return key.split("/", 1)[0]


@dataclass
class BatchJobReporter:
    backend: FireworksBatchBackend
    output_dir: Path
    # Maps a slice name such as 'dev_r1' to its examples (gold labels come from here).
    load_examples: Callable[[str], list[Example]]
    metrics: MetricsCalculator = field(default_factory=MetricsCalculator)
    parser: LabelParser = field(default_factory=LabelParser)
    # The prompt the batch was submitted with; decides which run directory it lands in.
    prompt_version: str = NliPromptV3.VERSION

    def report(self, job_id: str) -> dict[str, Any]:
        job = self.backend.get_job(job_id)
        state = job_state(job)
        if state != "COMPLETED":
            raise RuntimeError(f"job {job_id} is {state}; only COMPLETED jobs can be reported")

        run_dir = run_dir_for(self.output_dir, job["model"], self.prompt_version)
        job_dir = run_dir / "batch_jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        (job_dir / "job.json").write_text(json.dumps(job, indent=2))
        output_text = self._saved_or_fetch(job_dir / "output.jsonl", job["outputDatasetId"])
        if output_text is None:
            raise RuntimeError(f"could not download output dataset {job['outputDatasetId']}")
        input_text = self._saved_or_fetch(job_dir / "input.jsonl", job["inputDatasetId"])

        results = parse_batch_results(output_text)
        if input_text is not None:
            expected = [json.loads(line)["custom_id"] for line in input_text.splitlines() if line.strip()]
        else:
            # Without the input rows, the output rows are the only record of what was sent.
            expected = list(results)
        expected_set = set(expected)
        unexpected = [key for key in results if key not in expected_set]

        by_slice: dict[str, list[str]] = {}
        for key in expected:
            by_slice.setdefault(_slice_of(key), []).append(key)

        slice_metrics: dict[str, Any] = {}
        all_preds: list[Prediction] = []
        for slice_name, keys in by_slice.items():
            examples = {ex.key: ex for ex in self.load_examples(slice_name)}
            preds = [
                prediction_from_result(
                    examples[example_key_of(key)], key, results.get(key), self.parser
                )
                for key in keys
                if example_key_of(key) in examples
            ]
            PredictionStore(run_dir / f"{slice_name}.jsonl").upsert(preds)
            slice_metrics[slice_name] = self.metrics.compute(preds)
            all_preds.extend(preds)
        slice_metrics["overall"] = self.metrics.compute(all_preds)

        (job_dir / "metrics.json").write_text(json.dumps(slice_metrics, indent=2))
        summary = self._summary(
            job, job_id, self.prompt_version, expected, results, unexpected, slice_metrics, run_dir,
            input_available=input_text is not None,
        )
        (job_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    def _saved_or_fetch(self, path: Path, dataset: str) -> Optional[str]:
        """Return the saved artifact, else download it and save it. None if the API refuses."""
        if path.exists():
            return path.read_text()
        try:
            text = self.backend.download_dataset(dataset)
        except requests.HTTPError:
            return None
        path.write_text(text)
        return text

    @staticmethod
    def _summary(
        job: dict[str, Any],
        job_id: str,
        prompt_version: str,
        expected: list[str],
        results: dict[str, ChatResult],
        unexpected: list[str],
        slice_metrics: dict[str, Any],
        run_dir: Path,
        input_available: bool,
    ) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "job_name": job.get("name"),
            "model": job.get("model"),
            "prompt_version": prompt_version,
            "state": job_state(job),
            "create_time": job.get("createTime"),
            "lifecycle": job.get("lifecycle"),
            "input_dataset": job.get("inputDatasetId"),
            "output_dataset": job.get("outputDatasetId"),
            "job_progress": job.get("jobProgress"),
            "input_available": input_available,
            "requests_sent": len(expected),
            "rows_returned": len(results),
            "rows_with_errors": sum(1 for r in results.values() if r.error),
            "rows_missing": sum(1 for key in expected if key not in results),
            "unexpected_rows": len(unexpected),
            "metrics": {
                name: {
                    "n": m["n"],
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "errors": m["errors"],
                    "unparsed": m["unparsed"],
                    "pass_at_k": m["pass_at_k"],
                }
                for name, m in slice_metrics.items()
            },
            "predictions_dir": str(run_dir),
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process a completed Fireworks batch job.")
    parser.add_argument("--job-id", required=True, help="e.g. anli-job-6b9329a3a3 (or full name)")
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--prompt-version", choices=tuple(PROMPTS), default=NliPromptV3.VERSION,
        help="the prompt the batch job was submitted with",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    job_id = args.job_id.split("/")[-1]
    backend = FireworksBatchBackend(account_id=args.account_id)
    dataset = AnliDataset()

    def load_examples(slice_name: str) -> list[Example]:
        split, round_id = slice_name.split("_", 1)
        return dataset.load(split, round_id)

    reporter = BatchJobReporter(
        backend, Path(args.output_dir), load_examples, prompt_version=args.prompt_version
    )
    summary = reporter.report(job_id)
    for name, m in summary["metrics"].items():
        print(format_metrics_line(name, m))
    print(f"Wrote batch report to {summary['predictions_dir']}/batch_jobs/{job_id}/")


if __name__ == "__main__":
    main()
