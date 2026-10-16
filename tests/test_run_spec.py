"""Run identity: model slug, cached-result safety, sampling parity across request paths, deployment readiness."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anli_classification.config import RunConfig, model_slug, run_dir_for, split_model
from anli_classification.deploy import wait_until_serving
from anli_classification.llm.base import ChatRequest
from anli_classification.llm.fireworks import FireworksChatBackend, build_batch_jsonl
from anli_classification.run_spec import SPEC_FILE, RunSpec, RunSpecMismatch, ensure_run_spec

MODEL = "accounts/acme/models/anli-promoted"
DEPLOYMENT = "accounts/acme/deployments/anli-promoted-eval"


@pytest.mark.parametrize("name, slug", [
    ("accounts/fireworks/models/qwen3-4b", "qwen3-4b"),
    ("qwen3-4b", "qwen3-4b"),
    ("accounts/acme/models/anli-sft-step141", "anli-sft-step141"),
])
def test_plain_model_slugs_are_unchanged(name, slug):
    assert model_slug(name) == slug


def test_a_model_on_a_deployment_is_slugged_by_the_model_not_the_deployment():
    assert split_model(f"{MODEL}#{DEPLOYMENT}") == (MODEL, DEPLOYMENT)
    assert split_model(MODEL) == (MODEL, None)
    assert split_model(f"{MODEL}#") == (MODEL, None)
    assert model_slug(f"{MODEL}#{DEPLOYMENT}") == "anli-promoted" == model_slug(MODEL)


def test_run_dir_is_unchanged_without_a_tag_and_gains_a_suffix_with_one():
    base = run_dir_for(Path("out"), "accounts/fireworks/models/qwen3-4b", "zero-shot-v3", "server")
    assert base == Path("out/qwen3-4b/zero-shot-v3")
    assert run_dir_for(Path("out"), "x", "zero-shot-v3", "client") == Path("out/x/zero-shot-v3-client-template")
    assert run_dir_for(Path("out"), "x", "zero-shot-v3", "server", "t 0.0") == Path("out/x/zero-shot-v3--t_0.0")
    assert RunConfig(output_dir=Path("out"), run_tag="greedy").run_dir.name == "zero-shot-v3--greedy"


def _spec(**overrides):
    return RunSpec.from_config(RunConfig(**overrides))


def _cache(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "test_r3.jsonl").write_text(json.dumps({"key": "a"}) + "\n")


def test_a_new_folder_gets_its_spec_written_and_the_same_settings_match_later(tmp_path):
    spec = _spec(temperature=0.0)
    assert ensure_run_spec(tmp_path / "run", spec) == "new"
    assert json.loads((tmp_path / "run" / SPEC_FILE).read_text())["temperature"] == 0.0
    assert ensure_run_spec(tmp_path / "run", _spec(temperature=0.0)) == "match"


def test_a_run_at_temperature_0_after_one_at_0_6_is_refused_not_served_from_the_cache(tmp_path):
    run_dir = tmp_path / "run"
    ensure_run_spec(run_dir, _spec(temperature=0.6))
    _cache(run_dir)
    with pytest.raises(RunSpecMismatch) as error:
        ensure_run_spec(run_dir, _spec(temperature=0.0))
    assert error.value.differences == {"temperature": (0.6, 0.0)}
    assert "--run-tag" in str(error.value) and "temperature" in str(error.value)


def test_settings_that_do_not_change_answers_are_not_compared(tmp_path):
    run_dir = tmp_path / "run"
    ensure_run_spec(run_dir, _spec(backend="batch", concurrency=2))
    assert ensure_run_spec(run_dir, _spec(backend="chat", concurrency=32, limit=10, seed=5)) == "match"
    # the deployment is recorded but never compared
    with_deployment = _spec(model=f"{MODEL}#{DEPLOYMENT}")
    other = _spec(model=f"{MODEL}#another-deployment")
    assert with_deployment.model == MODEL and with_deployment.deployment == DEPLOYMENT
    assert with_deployment.differences(other) == {}


@pytest.mark.parametrize("change", [
    {"top_p": 1.0}, {"top_k": 20}, {"max_tokens": 8192}, {"samples": 3}, {"prompt_version": "zero-shot-v2"},
    {"template": "client"}, {"model": "accounts/acme/models/other"},
])
def test_each_setting_that_changes_answers_is_compared(change, tmp_path):
    ensure_run_spec(tmp_path / "run", _spec())
    _cache(tmp_path / "run")
    with pytest.raises(RunSpecMismatch):
        ensure_run_spec(tmp_path / "run", _spec(**change))


def test_an_older_folder_with_predictions_and_no_spec_is_reused_with_a_warning_status(tmp_path):
    run_dir = tmp_path / "run"
    _cache(run_dir)
    assert ensure_run_spec(run_dir, _spec(temperature=0.0)) == "legacy"
    assert not (run_dir / SPEC_FILE).exists()  # unknown settings are never guessed into a spec


def test_top_k_and_sampling_arguments_reach_all_three_request_paths():
    calls = {}

    def recorder(name):
        def create(**kwargs):
            calls[name] = kwargs
            choice = SimpleNamespace(message=SimpleNamespace(content="neutral"), text="neutral")
            return SimpleNamespace(choices=[choice])
        return create

    backend = FireworksChatBackend("m", api_key="k")
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=recorder("chat"))),
        completions=SimpleNamespace(create=recorder("completions")),
    )
    base = dict(key="k1", max_tokens=123, temperature=0.3, top_p=0.9, top_k=20)
    backend.complete([ChatRequest(messages=[{"role": "user", "content": "x"}], **base)])
    backend.complete([ChatRequest(messages=[], prompt="rendered", **base)])
    batch_body = json.loads(build_batch_jsonl([ChatRequest(messages=[{"role": "user", "content": "x"}], **base)]))["body"]

    def sampling(kwargs):
        extra = kwargs.get("extra_body", {})
        return (kwargs["max_tokens"], kwargs["temperature"], kwargs["top_p"], kwargs.get("top_k", extra.get("top_k")))

    assert sampling(calls["chat"]) == sampling(calls["completions"]) == sampling(batch_body) == (123, 0.3, 0.9, 20)

    calls.clear()  # no top_k requested: nothing extra is sent on any path
    unset = dict(base, top_k=None)
    backend.complete([ChatRequest(messages=[{"role": "user", "content": "x"}], **unset)])
    assert "extra_body" not in calls["chat"]


def test_wait_until_serving_retries_until_the_probe_succeeds_and_gives_up_after_the_timeout():
    answers = iter([False, OSError("not routed"), False, True])
    slept = []

    def probe():
        value = next(answers)
        if isinstance(value, Exception):
            raise value
        return value

    wait_until_serving(probe, timeout_s=100, interval_s=5, sleep=slept.append, clock=lambda: 0.0)
    assert slept == [5, 5, 5]

    now = [0.0]
    with pytest.raises(TimeoutError):
        wait_until_serving(lambda: False, timeout_s=30, interval_s=10, sleep=lambda s: now.__setitem__(0, now[0] + s),
                           clock=lambda: now[0])


def test_the_cli_refuses_a_cached_folder_made_at_another_temperature_before_any_model_call(tmp_path):
    from anli_classification.cli import main, parse_args

    first = ["--output-dir", str(tmp_path), "--temperature", "0.6", "--splits", "test", "--rounds", "r3"]
    config = parse_args(first)
    ensure_run_spec(config.run_dir, RunSpec.from_config(config))
    _cache(config.run_dir)
    with pytest.raises(SystemExit) as stop:
        main([*first[:2], "--temperature", "0.0", *first[4:]])
    assert "temperature" in str(stop.value) and "--run-tag" in str(stop.value)
    assert parse_args([*first, "--run-tag", "greedy"]).run_dir.name == "zero-shot-v3--greedy"
