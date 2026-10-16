"""Run guards: filter ratio, class drop, step cap, and a stall watchdog."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Sequence

from anli_classification.rl.stop import StopAtStep
from anli_classification.schemas import LABELS


def recent_filter_ratio(metrics_path: Path, window: int) -> Optional[float]:
    """Mean `rollout/filter_ratio` over the last `window` logged steps, or None until `window` steps exist."""
    if not metrics_path.exists():
        return None
    ratios = []
    for line in metrics_path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if "rollout/filter_ratio" in record:
                ratios.append(record["rollout/filter_ratio"])
    if len(ratios) < window:
        return None
    return sum(ratios[-window:]) / window


def print_eval(
    evaluation_fn: Any,
    max_steps: Optional[int] = None,
    stop_filter_ratio: Optional[float] = None,
    metrics_path: Optional[Path] = None,
    filter_window: int = 3,
    class_drop_alert: Optional[float] = None,
    stop_class_drop: Optional[float] = None,
) -> Any:
    """Print each dev evaluation as a JSON line, and stop training at `max_steps`.

    With `stop_filter_ratio`, it also stops once the mean filter ratio of the last `filter_window` steps
    exceeds it: most groups then have no reward variance, so further steps mostly pay for dropped rollouts.
    The check runs at evaluation steps only, so set `--eval-interval` to match `--checkpoint-interval`
    to have the stop step saved.

    With `class_drop_alert`, it prints an ALERT when a class's dev recall falls that far below the run's
    first evaluation (alert only). With `stop_class_drop`, a drop that large also stops the run, so the last
    checkpoint before a class collapses is the one kept.

    The cookbook sends evaluation metrics only to W&B and to COOKBOOK_METRICS_FILE, so without
    a W&B login they would be dropped silently. The stop happens here because the eval hook runs
    after the step's checkpoint is saved; the cookbook's cleanup still runs on the way out.
    """

    baseline: dict[str, float] = {}

    async def wrapped(step: int, rollout_fn: Any) -> dict[str, float]:
        metrics = await evaluation_fn(step, rollout_fn)
        print(json.dumps({"type": "eval", "step": step, **metrics}), flush=True)
        class_stop = False
        if class_drop_alert is not None or stop_class_drop is not None:
            for label in LABELS:
                key = f"dev/recall_{label}"
                if key not in metrics:
                    continue
                baseline.setdefault(key, metrics[key])
                drop = baseline[key] - metrics[key]
                if class_drop_alert is not None and drop > class_drop_alert:
                    print(f"ALERT: {label} dev recall {metrics[key]:.3f} is {drop:.3f} below the first evaluation "
                          f"({baseline[key]:.3f}) at step {step}", flush=True)
                if stop_class_drop is not None and drop > stop_class_drop:
                    print(f"Stopping at step {step}: {label} dev recall fell {drop:.3f} below the first evaluation "
                          f"(limit {stop_class_drop:.3f})", flush=True)
                    class_stop = True
        at_step_cap = max_steps is not None and step >= max_steps
        stop = class_stop or at_step_cap
        reason = "class_drop" if class_stop else ("max_steps" if at_step_cap else None)
        if not stop and stop_filter_ratio is not None and metrics_path is not None:
            ratio = recent_filter_ratio(metrics_path, filter_window)
            if ratio is not None and ratio > stop_filter_ratio:
                print(f"Stopping at step {step}: filter ratio {ratio:.0%} over the last {filter_window} steps "
                      f"exceeds {stop_filter_ratio:.0%}", flush=True)
                stop = True
                reason = "filter_ratio"
        if stop:
            from training.utils.logging import log_metrics

            log_metrics(metrics, step=step)  # the cookbook would log this after the hook returns
            raise StopAtStep(step, reason or "max_steps")
        return metrics

    return wrapped


class StallWatchdog:
    """Detects a run that has stopped making progress (e.g. hung connections) by watching its output files."""

    def __init__(self, paths: Sequence[Path], timeout: float, clock: Any = time.time) -> None:
        self.paths = list(paths)
        self.timeout = timeout
        self.clock = clock
        self.started = clock()

    def last_activity(self) -> float:
        mtimes = [p.stat().st_mtime for p in self.paths if p.exists()]
        return max([self.started, *mtimes])

    def stalled(self) -> bool:
        return self.clock() - self.last_activity() > self.timeout

    def watch(self, on_stall: Any, interval: float = 30.0) -> None:
        while not self.stalled():
            time.sleep(interval)
        on_stall()
