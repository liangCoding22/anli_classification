"""Per-rollout log and the malformed-answer tracker."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from anli_classification.rl.defaults import FORMAT_SNAPSHOT_EVERY


class FormatTracker:
    """Counts malformed training answers, writes a snapshot per window, and alerts on a high rate.

    Snapshots go into metrics.jsonl (records have `type: format` and no `step`, so the cookbook's
    per-step records are unaffected).
    """

    def __init__(self, metrics_path: Path, window: int = FORMAT_SNAPSHOT_EVERY, alert_rate: float = 0.05) -> None:
        self.metrics_path = metrics_path
        self.window = window
        self.alert_rate = alert_rate
        self.total = 0
        self.window_total = 0
        self.window_malformed = 0
        self._lock = threading.Lock()

    def record(self, well_formed: bool) -> None:
        with self._lock:
            self.total += 1
            self.window_total += 1
            if not well_formed:
                self.window_malformed += 1
            if self.total % self.window == 0:
                self._flush_window()

    def flush(self) -> None:
        with self._lock:
            if self.window_total:
                self._flush_window()

    def _flush_window(self) -> None:  # caller holds the lock
        rate = self.window_malformed / self.window_total
        record = {
            "type": "format",
            "rollouts_total": self.total,
            "window_rollouts": self.window_total,
            "format/malformed": self.window_malformed,
            "format/malformed_rate": rate,
        }
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        if rate > self.alert_rate:
            print(
                f"ALERT: {rate:.1%} of the last {self.window_total} training answers are malformed "
                f"(not exactly one label word; threshold {self.alert_rate:.0%})",
                flush=True,
            )
        self.window_total = 0
        self.window_malformed = 0


class RolloutLog:
    """Appends one JSON line per graded rollout, from any thread."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record) + "\n"
        with self._lock, self.path.open("a") as fh:
            fh.write(line)
