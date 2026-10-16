"""Named recipes: backward compatible by default, and the v2 recipe matches the run it names."""

import json
from pathlib import Path

import pytest

from anli_classification.rl.launch import code_version, write_run_config
from anli_classification.rl.recipes import RECIPES
from anli_classification.rl_train import parse_args

BASE = ["--pool-path", "pool.jsonl"]
RECORDED = Path(__file__).resolve().parent.parent / "reproducibility" / "runs" / "n50000-v2" / "run_config.json"


def test_without_a_recipe_the_original_defaults_apply():
    args = parse_args(BASE)
    assert args.recipe == "legacy"
    assert (args.completions_per_prompt, args.prompt_groups_per_step, args.learning_rate) == (8, 64, 1e-5)
    assert (args.checkpoint_interval, args.eval_interval, args.max_steps) == (2, 5, None)
    assert (args.consistency_penalty, args.stop_class_drop, args.stop_filter_ratio, args.skip_cleanup) == (0.0, None, None, False)
    assert args.reward_version == "v2"


def test_the_v2_recipe_sets_every_flag_of_the_last_run():
    args = parse_args([*BASE, "--recipe", "v2"])
    for name, value in RECIPES["v2"].items():
        assert getattr(args, name) == value, name
    assert args.recipe == "v2" and args.pool_path == "pool.jsonl"


def test_explicit_flags_win_over_the_recipe():
    args = parse_args([*BASE, "--recipe", "v2", "--learning-rate", "1e-5", "--no-skip-cleanup", "--max-steps", "10",
                       "--eval-interval", "5", "--checkpoint-interval", "5"])
    assert args.learning_rate == 1e-5 and args.skip_cleanup is False and args.max_steps == 10
    assert args.completions_per_prompt == 16  # untouched recipe values stay


def test_the_old_skip_cleanup_flag_still_works():
    assert parse_args([*BASE, "--skip-cleanup"]).skip_cleanup is True


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_every_recipe_key_is_a_real_flag(recipe):
    known = set(vars(parse_args(BASE)))
    assert set(RECIPES[recipe]) <= known


@pytest.mark.skipif(not RECORDED.exists(), reason="the recorded v2 run config is not present")
def test_the_v2_recipe_matches_what_the_recorded_run_actually_used():
    recorded = json.loads(RECORDED.read_text())["args"]
    missing = []
    for name, value in RECIPES["v2"].items():
        if name not in recorded:
            missing.append(name)
            continue
        assert recorded[name] == str(value), f"{name}: the recorded run used {recorded[name]}, the recipe says {value}"
    assert missing == ["reward_version"]  # the v2 run predates the flag; the recipe pins the reward it used


def test_run_config_records_the_code_version(tmp_path):
    version = code_version()
    assert set(version) == {"git_sha", "dirty"}
    assert version["git_sha"] is None or len(version["git_sha"]) == 40
    write_run_config(parse_args(BASE), tmp_path, [{"key": "a"}], [{"key": "b"}])
    record = json.loads((tmp_path / "run_config.json").read_text())
    assert record["code"] == version or record["code"]["git_sha"] == version["git_sha"]
    assert record["args"]["recipe"] == "legacy"
