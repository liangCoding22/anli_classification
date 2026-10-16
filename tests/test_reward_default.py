"""v2 is the reward every reported result used, and it stays the default until v3 has been tried in a training run."""

import inspect

import pytest

from anli_classification.rl.cli import parse_args
from anli_classification.rl.recipes import RECIPES
from anli_classification.rl.reward import training_reward
from anli_classification.rl.rollout import make_rollout_fn


def test_v2_is_the_default_everywhere_the_reward_version_is_chosen():
    assert parse_args(["--pool-path", "x"]).reward_version == "v2"
    assert parse_args(["--pool-path", "x", "--recipe", "v2"]).reward_version == "v2"
    assert parse_args(["--pool-path", "x", "--recipe", "legacy"]).reward_version == "v2"
    assert inspect.signature(make_rollout_fn).parameters["reward_version"].default == "v2"
    assert inspect.signature(training_reward).parameters["version"].default == "v2"
    assert RECIPES["v2"]["reward_version"] == "v2"


def test_choosing_v3_is_always_explicit():
    assert parse_args(["--pool-path", "x", "--reward-version", "v3"]).reward_version == "v3"
    with pytest.raises(SystemExit):
        parse_args(["--pool-path", "x", "--reward-version", "v4"])


@pytest.mark.parametrize("label, pred, verdict, expected", [
    (1.0, 1, "consistent", 1.0),
    (1.0, 1, "inconsistent", 0.75),
    (0.0, 1, "consistent", 0.0),
    (0.0, 1, "inconsistent", -0.25),
    (0.0, None, None, 0.0),  # malformed scores 0 in v2 (the gap v3 closes)
    (1.0, 1, None, 1.0),  # no verdict (no reasoning, or a judge outage) is never penalised in v2
    (1.0, 1, "unclear", 1.0),
])
def test_the_v2_reward_values_are_pinned_for_a_penalty_of_0_25(label, pred, verdict, expected):
    assert training_reward(label, pred, verdict, 0.25, "v2") == expected
    assert training_reward(label, pred, verdict, 0.25) == expected  # the default version is v2
