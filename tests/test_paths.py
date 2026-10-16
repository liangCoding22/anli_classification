from pathlib import Path

from anli_classification import paths


def test_default_paths_are_the_ones_the_scripts_used_before(monkeypatch):
    monkeypatch.delenv("ANLI_OUTPUT_DIR", raising=False)
    assert paths.output_root() == Path("outputs")
    assert paths.default_pool_path() == Path("outputs/qwen3-4b/zero-shot-v3/training_pool/train_r3_n50000_seed0_excl/pool_mixed.jsonl")
    assert paths.training_pool_root() == Path("outputs/qwen3-4b/zero-shot-v3/training_pool")
    assert paths.teacher_traces_glob() == "outputs/teacher_traces/*/*/traces.jsonl"
    assert paths.teacher_prompts_glob() == "outputs/teacher_prompts/*.jsonl"


def test_the_output_root_can_be_moved_with_one_environment_variable(monkeypatch):
    monkeypatch.setenv("ANLI_OUTPUT_DIR", "/data/anli")
    assert paths.default_pool_path().parts[:3] == ("/", "data", "anli")
    assert paths.teacher_traces_glob().startswith("/data/anli/teacher_traces/")
