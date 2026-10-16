"""How a run is ended from inside the cookbook's loop."""

from __future__ import annotations


class StopAtStep(BaseException):
    """Ends training after the requested step. A BaseException, so the cookbook's eval wrapper does not catch it."""

    def __init__(self, step: int) -> None:
        super().__init__(f"stopped at step {step}")
        self.step = step
