"""The behaviour of the installed cookbook that rl/ relies on. Skipped when the cookbook is not importable.

Run where the cookbook is installed (Python 3.11+, PYTHONPATH containing the cookbook root):
    PYTHONPATH=/path/to/cookbook:src python -m pytest tests/test_cookbook_contract.py
"""

import asyncio
import inspect

import pytest

from anli_classification.rl.guards import print_eval
from anli_classification.rl.stop import StopAtStep

heldout = pytest.importorskip("training.utils.rl.heldout", reason="the Fireworks cookbook is not importable here")


def _run(evaluation_fn):
    async def rollout_fn(row, evaluation=False):
        return None

    return asyncio.run(heldout.run_evaluation(2, evaluation_fn, rollout_fn))


def test_stop_at_step_propagates_through_the_cookbooks_evaluation_runner():
    async def evaluation(step, rollout_fn):
        return {"dev/recall_neutral": 0.5}

    with pytest.raises(StopAtStep) as stop:
        _run(print_eval(evaluation, max_steps=2))
    assert stop.value.step == 2


def test_an_ordinary_exception_in_the_evaluation_is_swallowed_which_is_why_the_stop_is_a_base_exception():
    async def broken(step, rollout_fn):
        raise ValueError("boom")

    metrics = _run(broken)
    assert metrics["eval/failed"] == 1.0


def test_the_cookbook_loop_still_accepts_the_hooks_the_driver_passes():
    rl_loop = pytest.importorskip("training.recipes.rl_loop", reason="the cookbook loop needs its full dependencies")
    parameters = inspect.signature(rl_loop.main).parameters
    for name in ("rollout_fn_factory", "rows", "evaluation_fn", "evaluation_interval", "advantage_fn"):
        assert name in parameters, f"rl_loop.main no longer takes {name}"


def test_the_checkpoint_save_signature_the_patch_wraps_is_unchanged():
    checkpoints = pytest.importorskip("training.utils.checkpoints", reason="the cookbook checkpoints need their dependencies")
    parameters = inspect.signature(checkpoints.TrainingCheckpoints.save).parameters
    for name in ("name", "resumable", "promotable", "data_consumed"):
        assert name in parameters
