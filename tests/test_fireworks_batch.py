import json

from anli_classification.llm.base import ChatRequest, ChatResult
from anli_classification.llm.fireworks import build_batch_jsonl, parse_batch_results


def test_build_batch_jsonl_one_row_per_request():
    reqs = [
        ChatRequest("dev_r1/a", [{"role": "user", "content": "hi"}], 64),
        ChatRequest("dev_r1/b", [{"role": "user", "content": "yo"}], 64),
    ]
    rows = [json.loads(line) for line in build_batch_jsonl(reqs).splitlines()]
    assert [r["custom_id"] for r in rows] == ["dev_r1/a", "dev_r1/b"]
    assert rows[0]["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert rows[0]["body"]["max_tokens"] == 64


def test_parse_batch_results_success_and_error_rows():
    text = "\n".join(
        [
            json.dumps({"custom_id": "a", "response": {"choices": [{"message": {"content": "entailment"}}]}}),
            json.dumps({"custom_id": "b", "error": {"message": "bad request"}}),
            json.dumps({"custom_id": "c", "response": {"choices": []}}),
        ]
    )
    results = parse_batch_results(text)
    assert results["a"] == ChatResult("a", "entailment", None)
    assert results["b"].error and results["b"].text == ""
    assert results["c"].error == "no choices in batch response"


def test_batch_complete_saves_input_job_and_output_artifacts(tmp_path, monkeypatch):
    from anli_classification.llm.fireworks import FireworksBatchBackend

    backend = FireworksBatchBackend("m", account_id="acct", api_key="test", artifacts_dir=tmp_path / "batch_jobs")
    output = json.dumps(
        {"custom_id": "a", "response": {"choices": [{"message": {"content": "neutral"}}]}}
    )
    monkeypatch.setattr(backend, "_create_dataset", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_upload", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_create_job", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_wait", lambda job_id: {"state": "JOB_STATE_COMPLETED"})
    monkeypatch.setattr(backend, "download_dataset", lambda output_id: output)

    results = backend.complete([ChatRequest("a", [{"role": "user", "content": "hi"}], 8)])
    assert results[0].text == "neutral"

    job_dirs = list((tmp_path / "batch_jobs").iterdir())
    assert len(job_dirs) == 1
    names = sorted(p.name for p in job_dirs[0].iterdir())
    assert names == ["input.jsonl", "job.json", "output.jsonl"]
    assert (job_dirs[0] / "output.jsonl").read_text() == output


def test_failed_batch_job_still_saves_job_record(tmp_path, monkeypatch):
    import pytest
    from anli_classification.llm.fireworks import FireworksBatchBackend

    backend = FireworksBatchBackend("m", account_id="acct", api_key="test", artifacts_dir=tmp_path)
    monkeypatch.setattr(backend, "_create_dataset", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_upload", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_create_job", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_wait", lambda job_id: {"state": "JOB_STATE_FAILED"})

    with pytest.raises(RuntimeError, match="FAILED"):
        backend.complete([ChatRequest("a", [], 8)])
    job_dirs = list(tmp_path.iterdir())
    assert len(job_dirs) == 1 and (job_dirs[0] / "job.json").exists()
