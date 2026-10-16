"""How a run is ended from inside the cookbook's loop."""

from __future__ import annotations


class StopAtStep(BaseException):
    """Ends training after the requested step. A BaseException, so the cookbook's eval wrapper does not catch it.

    The cookbook has no stop hook, and `run_evaluation` turns any `Exception` raised by the evaluation function into
    `eval/failed = 1`, so this is the one way to end a run from there. `tests/test_cookbook_contract.py` checks that it
    still propagates through the installed cookbook. `reason` says which guard ended the run: "max_steps", "filter_ratio"
    or "class_drop".
    """

    def __init__(self, step: int, reason: str = "max_steps") -> None:
        super().__init__(f"stopped at step {step}")
        self.step = step
        self.reason = reason
