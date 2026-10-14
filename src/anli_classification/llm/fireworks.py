"""Fireworks backends.

- `FireworksChatBackend`: concurrent chat-completions calls. Low latency; use for
  small runs and debugging.
- `FireworksBatchBackend`: Fireworks Batch Inference API (upload JSONL, create a
  job, poll, download). About half the serverless per-token price; use for full runs.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

import requests
from openai import OpenAI

from anli_classification.llm.base import ChatRequest, ChatResult
from anli_classification.parsers import split_reasoning

FIREWORKS_INFERENCE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_API_URL = "https://api.fireworks.ai/v1"
TERMINAL_JOB_STATES = {"COMPLETED", "FAILED", "EXPIRED"}


def job_state(job: dict[str, Any]) -> str:
    """API reports states as e.g. 'JOB_STATE_COMPLETED'; return the bare name."""
    return str(job.get("state", "")).replace("JOB_STATE_", "", 1)


def _resolve_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise RuntimeError("Set FIREWORKS_API_KEY (or pass api_key) to call Fireworks.")
    return key


class FireworksChatBackend:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_workers: int = 8,
        max_retries: int = 5,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.max_workers = max_workers
        self._client = OpenAI(
            base_url=FIREWORKS_INFERENCE_URL,
            api_key=_resolve_api_key(api_key),
            max_retries=max_retries,
            timeout=timeout,
        )

    def complete(self, chat_requests: Sequence[ChatRequest]) -> list[ChatResult]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._complete_one, chat_requests))

    def _complete_one(self, request: ChatRequest) -> ChatResult:
        if request.prompt is not None:
            return self._complete_rendered(request)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=list(request.messages),
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            message = response.choices[0].message
            # Fireworks reasoning models return thinking in a separate field.
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
            return ChatResult(
                request.key,
                message.content or "",
                reasoning=reasoning or "",
            )
        except Exception as exc:  # SDK already retried transient failures
            return ChatResult(request.key, "", f"{type(exc).__name__}: {exc}")

    def _complete_rendered(self, request: ChatRequest) -> ChatResult:
        """A client-rendered prompt goes to /completions, so no server chat template applies."""
        try:
            response = self._client.completions.create(
                model=self.model,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                **({"extra_body": {"top_k": request.top_k}} if request.top_k is not None else {}),
            )
            reasoning, text = split_reasoning(response.choices[0].text or "")
            return ChatResult(request.key, text, reasoning=reasoning)
        except Exception as exc:  # SDK already retried transient failures
            return ChatResult(request.key, "", f"{type(exc).__name__}: {exc}")


def build_batch_jsonl(chat_requests: Sequence[ChatRequest]) -> str:
    """One batch input row per request; the model is set on the job, not per row."""
    return "\n".join(
        json.dumps(
            {
                "custom_id": request.key,
                "body": {
                    "messages": list(request.messages),
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    **({"top_k": request.top_k} if request.top_k is not None else {}),
                },
            }
        )
        for request in chat_requests
    )


def parse_batch_results(text: str) -> dict[str, ChatResult]:
    """Parse batch output JSONL (success and error rows) into results keyed by custom_id."""
    results: dict[str, ChatResult] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("custom_id", ""))
        if row.get("error"):
            results[key] = ChatResult(key, "", json.dumps(row["error"]))
            continue
        # Row shape from a real job: {"custom_id", "response": {"choices": [...]}}.
        choices = (row.get("response") or row).get("choices") or []
        if not choices:
            results[key] = ChatResult(key, "", "no choices in batch response")
            continue
        message = choices[0]["message"]
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if not reasoning:
            # A model served without a reasoning parser returns its thinking inline.
            reasoning, text = split_reasoning(text)
        results[key] = ChatResult(key, text, reasoning=reasoning)
    return results


class FireworksBatchBackend:
    """Runs each `complete()` call as one Fireworks batch job.

    If `artifacts_dir` is set, each job's files are kept under `artifacts_dir/<job_id>/`:
    `input.jsonl` (requests sent), `job.json` (final job record), and `output.jsonl`
    (raw response rows as returned by Fireworks).
    """

    def __init__(
        self,
        model: str = "",  # only needed for complete(); reporting works without it
        account_id: str = "",
        api_key: Optional[str] = None,
        poll_seconds: float = 30.0,
        artifacts_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.account_id = account_id
        self.poll_seconds = poll_seconds
        self.artifacts_dir = artifacts_dir
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {_resolve_api_key(api_key)}"

    def complete(self, chat_requests: Sequence[ChatRequest]) -> list[ChatResult]:
        if not chat_requests:
            return []
        suffix = uuid.uuid4().hex[:10]
        input_id, output_id, job_id = f"anli-in-{suffix}", f"anli-out-{suffix}", f"anli-job-{suffix}"
        input_jsonl = build_batch_jsonl(chat_requests)

        self._create_dataset(input_id, example_count=len(chat_requests))
        self._upload(input_id, input_jsonl)
        self._create_job(job_id, input_id, output_id)
        self._save_artifact(job_id, "input.jsonl", input_jsonl)
        job = self._wait(job_id)
        self._save_artifact(job_id, "job.json", json.dumps(job, indent=2))
        if job_state(job) != "COMPLETED":
            raise RuntimeError(f"batch job {job_id} ended in state {job_state(job)}")

        text = self.download_dataset(output_id)
        self._save_artifact(job_id, "output.jsonl", text)
        results = parse_batch_results(text)
        return [
            results.get(r.key, ChatResult(r.key, "", "missing from batch output"))
            for r in chat_requests
        ]

    def _save_artifact(self, job_id: str, name: str, content: str) -> None:
        if self.artifacts_dir is None:
            return
        job_dir = self.artifacts_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / name).write_text(content)

    # --- HTTP helpers -----------------------------------------------------

    def _account_url(self, path: str) -> str:
        return f"{FIREWORKS_API_URL}/accounts/{self.account_id}/{path}"

    def _create_dataset(self, dataset_id: str, example_count: int) -> None:
        self._post(
            self._account_url("datasets"),
            json={
                "datasetId": dataset_id,
                "dataset": {"userUploaded": {}, "exampleCount": example_count},
            },
        )

    def _upload(self, dataset_id: str, jsonl: str) -> None:
        url = self._account_url(f"datasets/{dataset_id}:upload")
        response = self._session.post(url, files={"file": ("input.jsonl", jsonl)})
        response.raise_for_status()

    def _create_job(self, job_id: str, input_id: str, output_id: str) -> None:
        account = f"accounts/{self.account_id}"
        self._post(
            self._account_url("batchInferenceJobs"),
            params={"batchInferenceJobId": job_id},
            json={
                "model": self.model,
                "inputDatasetId": f"{account}/datasets/{input_id}",
                "outputDatasetId": f"{account}/datasets/{output_id}",
            },
        )

    def _wait(self, job_id: str) -> dict[str, Any]:
        url = self._account_url(f"batchInferenceJobs/{job_id}")
        while True:
            job = self._session.get(url).json()
            if job_state(job) in TERMINAL_JOB_STATES:
                return job
            time.sleep(self.poll_seconds)

    def get_job(self, job_id: str) -> dict[str, Any]:
        response = self._session.get(self._account_url(f"batchInferenceJobs/{job_id}"))
        response.raise_for_status()
        return response.json()

    def download_dataset(self, dataset: str) -> str:
        """Download all files of a dataset; accepts a bare id or 'accounts/.../datasets/<id>'."""
        dataset_id = dataset.split("/")[-1]
        url = self._account_url(f"datasets/{dataset_id}:getDownloadEndpoint")
        response = self._session.get(url)
        response.raise_for_status()
        signed = response.json().get("filenameToSignedUrls", {})
        return "\n".join(requests.get(link).text for link in signed.values())

    def _post(self, url: str, **kwargs: Any) -> dict[str, Any]:
        response = self._session.post(url, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}
