"""Runtime patches to the cookbook's training library. Each is idempotent and leaves the installed files alone."""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any

from anli_classification.rl.defaults import TRAINER_JOB_FILE


_INTERMEDIATE_NAME = re.compile(r"step-\d+")


def is_intermediate_checkpoint(name: str) -> bool:
    """Periodic checkpoints are named step-N; the final one is saved separately and already promotable."""
    return _INTERMEDIATE_NAME.fullmatch(name) is not None


HOTLOAD_REQUEST_TIMEOUT_S = 900  # the SDK's default is 60 s, too short for the first base-snapshot hotload


def install_hotload_timeout_patch() -> None:
    """Raise the default timeout of DeploymentManager.hotload, which hotload_and_wait never overrides.

    Without this, the first weight sync (a full base snapshot) times out after 60 s with a ReadTimeout
    and the run fails after its trainer and deployment are already running. The installed SDK is not modified.
    """
    from fireworks.training.sdk.deployment import DeploymentManager

    if getattr(DeploymentManager.hotload, "_anli_timeout_patched", False):
        return
    original = DeploymentManager.hotload

    @functools.wraps(original)
    def hotload(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", HOTLOAD_REQUEST_TIMEOUT_S)
        return original(self, *args, **kwargs)

    hotload._anli_timeout_patched = True
    DeploymentManager.hotload = hotload


def install_checkpoint_patches(log_dir: Path, promotable_checkpoints: bool) -> None:
    """Runtime patches on the cookbook's TrainingCheckpoints. The installed cookbook files are not modified.

    - Writes the trainer job id to `log_dir/trainer_job.json` when the trainer is created.
    - With `promotable_checkpoints`, each intermediate save (step-N) also saves a promotable
      base sampler snapshot, so it can be promoted to a model after the job finishes.
      The cookbook's periodic save passes promotable=False; this overrides only that.
    """
    from training.utils import checkpoints as cp

    if getattr(cp.TrainingCheckpoints, "_anli_patched", False):
        return
    original_init = cp.TrainingCheckpoints.__init__
    original_save = cp.TrainingCheckpoints.save

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        trainer_id = kwargs.get("trainer_id")
        if trainer_id:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / TRAINER_JOB_FILE).write_text(json.dumps({"trainer_job_id": trainer_id}, indent=2))

    def save(self: Any, name: str, *, resumable: bool, promotable: bool, data_consumed: Any = None) -> None:
        if promotable_checkpoints and resumable and not promotable and is_intermediate_checkpoint(name):
            promotable = True
        return original_save(self, name, resumable=resumable, promotable=promotable, data_consumed=data_consumed)

    cp.TrainingCheckpoints.__init__ = init
    cp.TrainingCheckpoints.save = save
    cp.TrainingCheckpoints._anli_patched = True
