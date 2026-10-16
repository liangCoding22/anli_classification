"""Wait until a freshly created deployment actually answers requests.

A deployment can report READY before requests to `<model>#<deployment>` are routed to it: the first requests then fail
with 404 "model not found". In one 1,200-example evaluation 8 requests failed that way. A one-token probe, retried until it
succeeds, avoids sending the evaluation into that window.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import requests

from anli_classification.llm.fireworks import FIREWORKS_INFERENCE_URL


def serving_probe(model: str, api_key: Optional[str] = None, base_url: str = FIREWORKS_INFERENCE_URL) -> Callable[[], bool]:
    """A function that sends a one-token request to `model` and says whether it was served."""
    key = api_key or os.environ["FIREWORKS_API_KEY"]

    def probe() -> bool:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1, "temperature": 0},
            timeout=60,
        )
        return response.status_code == 200

    return probe


def wait_until_serving(
    probe: Callable[[], bool],
    timeout_s: float = 600.0,
    interval_s: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Call `probe` until it returns True; an exception counts as not ready. Raises TimeoutError after `timeout_s`."""
    start = clock()
    while True:
        try:
            if probe():
                return
        except Exception:
            pass
        if clock() - start >= timeout_s:
            raise TimeoutError(f"the deployment did not answer a test request within {timeout_s:.0f}s")
        sleep(interval_s)
