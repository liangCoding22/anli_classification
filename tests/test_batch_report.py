import json

import pytest

from anli_classification.batch_report import BatchJobReporter
from anli_classification.prompts import NliPromptV3
from anli_classification.schemas import Example


class FakeBatchBackend:
    def __init__(self, job, datasets):
        self.job = job
        self.datasets = datasets  # dataset id -> text

    def get_job(self, job_id):
        return self.job

    def download_dataset(self, dataset):
        return self.datasets[dataset.split("/")[-1]]


def row(key, text):
    return json.dumps({"custom_id": key, "response": {"choices": [{"message": {"content": text}}]}})


def test_report_writes_predictions_metrics_and_summary(tmp_path):
    examples = [
        Example("dev", "r1", "a", "p", "h", 0),
        Example("dev", "r1", "b", "p", "h", 1),
        Example("dev", "r1", "c", "p", "h", 2),
    ]
    input_text = "\n".join(
        json.dumps({"custom_id": e.key, "body": {}}) for e in examples
    )
    # 'c' is missing from the output, so it should be recorded as a failed prediction.
    output_text = "\n".join([row("dev_r1/a", "entailment"), row("dev_r1/b", "contradiction")])
    job = {
        "name": "accounts/acct/batchInferenceJobs/job1",
        "model": "accounts/fireworks/models/demo-model",
        "state": "JOB_STATE_COMPLETED",
        "inputDatasetId": "accounts/acct/datasets/in1",
        "outputDatasetId": "accounts/acct/datasets/out1",
        "jobProgress": {"successfullyProcessedRequests": 2},
    }
    backend = FakeBatchBackend(job, {"in1": input_text, "out1": output_text})
    reporter = BatchJobReporter(backend, tmp_path, lambda name: examples)

    summary = reporter.report("job1")

    run_dir = tmp_path / "demo-model" / NliPromptV3.VERSION
    job_dir = run_dir / "batch_jobs" / "job1"
    assert {p.name for p in job_dir.iterdir()} == {
        "job.json", "input.jsonl", "output.jsonl", "metrics.json", "summary.json",
    }
    preds = [json.loads(line) for line in (run_dir / "dev_r1.jsonl").read_text().splitlines()]
    assert {p["key"]: p["pred"] for p in preds} == {"dev_r1/a": 0, "dev_r1/b": 2, "dev_r1/c": None}
    assert summary["requests_sent"] == 3 and summary["rows_missing"] == 1
    assert summary["metrics"]["overall"]["n"] == 3
    assert summary["metrics"]["overall"]["accuracy"] == pytest.approx(1 / 3)  # only "a" is right


def test_report_refuses_jobs_that_are_not_completed(tmp_path):
    job = {"name": "x", "model": "m", "state": "JOB_STATE_RUNNING"}
    reporter = BatchJobReporter(FakeBatchBackend(job, {}), tmp_path, lambda name: [])
    with pytest.raises(RuntimeError, match="RUNNING"):
        reporter.report("job1")
