"""Deleting what a run created, so a failed or stalled run stops billing."""

from __future__ import annotations

import argparse
import functools
import json
import os
import threading
import urllib.request
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Optional

from anli_classification.rl.defaults import METRICS_FILE, ROLLOUTS_FILE, TRAINER_JOB_FILE
from anli_classification.rl.guards import StallWatchdog


def _fireworks_api(method: str, path: str, api_key: str, base_url: str) -> Any:
    request = urllib.request.Request(f"{base_url}/v1/{path}", method=method, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    return json.loads(body) if body else {}


def delete_run_resources(
    log_dir: Path, started_at: float, base_model: str, api: Any = _fireworks_api, deployment_id: Optional[str] = None
) -> list[str]:
    """Delete the trainer job and the deployment this run created, so a crashed run stops billing.

    The cookbook only scales the deployment to zero on exit, and does nothing under --skip-cleanup. The trainer
    comes from `trainer_job.json` (or, when that was never written, any trainer on `base_model` created after
    `started_at`); deployments are those on `base_model` created after `started_at`,
    plus `deployment_id` when the run reattached to an older one.
    Every step is best effort and reported, so a failed delete never hides the original error.
    """
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    call = functools.partial(api, api_key=api_key, base_url=base_url)
    since = datetime.fromtimestamp(started_at, timezone.utc) - timedelta(minutes=2)
    done: list[str] = []

    def delete(path: str) -> None:
        try:
            call("DELETE", path)
            done.append(f"deleted {path}")
        except Exception as exc:  # already gone or refused: report and carry on
            done.append(f"could not delete {path}: {exc}")

    account = call("GET", "accounts")["accounts"][0]["name"]
    job_file = log_dir / TRAINER_JOB_FILE
    if job_file.exists():
        delete(f"{account}/rlorTrainerJobs/{json.loads(job_file.read_text())['trainer_job_id']}")
    else:  # the file is written once the trainer is up, so a run that never got a GPU has no record of its job
        for job in call("GET", f"{account}/rlorTrainerJobs?pageSize=100").get("rlorTrainerJobs", []):
            created = datetime.fromisoformat(job["createTime"].replace("Z", "+00:00"))
            if (job.get("state") != "JOB_STATE_DELETED" and created >= since
                    and job.get("trainingConfig", {}).get("baseModel") == base_model):
                delete(job["name"])
    if deployment_id:
        delete(f"{account}/deployments/{deployment_id}?ignoreChecks=true")
    for deployment in call("GET", f"{account}/deployments?pageSize=100").get("deployments", []):
        if deployment_id and deployment["name"].endswith(f"/{deployment_id}"):
            continue
        created = datetime.fromisoformat(deployment["createTime"].replace("Z", "+00:00"))
        if deployment.get("baseModel") == base_model and created >= since:
            delete(f"{deployment['name']}?ignoreChecks=true")
    return done


def start_stall_watchdog(args: argparse.Namespace, log_dir: Path, started_at: float) -> None:
    """Background thread: on a stall, delete the run's cloud resources and end the process."""
    if not args.stall_timeout:
        return

    def on_stall() -> None:
        print(f"STALL: no rollouts or metrics for {args.stall_timeout}s; stopping the run", flush=True)
        if args.delete_on_failure:
            try:
                for line in delete_run_resources(log_dir, started_at, args.base_model, deployment_id=args.deployment_id):
                    print(f"STALL CLEANUP: {line}", flush=True)
            except Exception as error:
                print(f"STALL CLEANUP FAILED, delete the trainer and deployment yourself: {error}", flush=True)
        os._exit(1)  # the main thread is stuck inside the cookbook loop, so it cannot be interrupted cleanly

    watchdog = StallWatchdog([log_dir / ROLLOUTS_FILE, log_dir / METRICS_FILE], args.stall_timeout)
    threading.Thread(target=watchdog.watch, args=(on_stall,), daemon=True).start()
