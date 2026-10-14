import json

from types import SimpleNamespace

from anli_classification.rl_train import dev_accuracy, label_reward, load_pool_rows


def test_label_reward_is_one_only_for_the_gold_label():
    assert label_reward("entailment", 0) == 1.0
    assert label_reward("Neutral", 1) == 1.0
    assert label_reward("Neutral.", 1) == 0.0  # trailing punctuation is not allowed
    assert label_reward("contradiction", 0) == 0.0
    assert label_reward("I am not sure", 0) == 0.0


def test_load_pool_rows_builds_messages_and_gold(tmp_path):
    record = {
        "key": "train_r3/u1",
        "group": "1-3/4",
        "gold": 2,
        "premise": "A man sleeps.",
        "hypothesis": "A man is awake.",
        "reason": "",
        "n_correct": 3,
        "samples": [],
    }
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps(record) + "\n")
    rows = load_pool_rows(pool)
    assert len(rows) == 1
    assert rows[0]["gold"] == 2 and rows[0]["key"] == "train_r3/u1"
    assert rows[0]["messages"][0]["role"] == "system"
    assert "A man sleeps." in rows[0]["messages"][1]["content"]


def run(reward):
    return SimpleNamespace(segments=[SimpleNamespace(reward=reward)])


def test_dev_accuracy_counts_failed_rollouts_as_incorrect():
    rows = [{"key": "a"}, {"key": "b"}, {"key": "c"}, {"key": "d"}]
    runs_by_row = [[run(1.0)], [run(0.0)], [None], [run(1.0)]]  # one failed rollout
    metrics = dev_accuracy(rows, runs_by_row)
    assert metrics["dev/accuracy"] == 2 / 4  # failed row is in the denominator
    assert metrics["dev/valid_accuracy"] == 2 / 3  # over the three scored rollouts
    assert metrics["dev/dropped_ratio"] == 1 / 4


def test_dev_accuracy_with_no_failures_matches_both_metrics():
    rows = [{"key": "a"}, {"key": "b"}]
    metrics = dev_accuracy(rows, [[run(1.0)], [run(0.0)]])
    assert metrics["dev/accuracy"] == metrics["dev/valid_accuracy"] == 0.5
    assert metrics["dev/dropped_ratio"] == 0.0


def test_label_reward_missing_answer_is_zero():
    assert label_reward("", 0) == 0.0
    assert label_reward("The premise says nothing useful", 1) == 0.0


def test_label_reward_truncated_reasoning_without_label_is_zero():
    truncated = "Let me think. The hypothesis could follow from the premise if we assume"
    assert label_reward(truncated, 0) == 0.0


def test_label_reward_requires_exactly_one_label_in_the_answer():
    assert label_reward("neutral", 1) == 1.0
    assert label_reward("Entailment. Not neutral.", 0) == 0.0  # not exactly one label word
    assert label_reward("neutral\n\nexplanation after the label", 1) == 0.0
    assert label_reward("<think>entailment</think>neutral", 1) == 1.0
    assert label_reward("<think>neutral, but unclosed", 1) == 0.0


def test_thinking_text_extracts_only_reasoning_parts():
    from anli_classification.rl_train import thinking_text

    message = {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "maybe neutral"},
        {"type": "text", "text": "entailment"},
    ]}
    assert thinking_text(message) == "maybe neutral"
    assert thinking_text({"role": "assistant", "content": "entailment"}) == ""


def test_is_well_formed_accepts_only_one_label_word():
    from anli_classification.parsers import is_well_formed

    assert is_well_formed("entailment")
    assert is_well_formed("  Neutral\n")  # case and surrounding whitespace are ignored
    assert is_well_formed("CONTRADICTION")
    assert not is_well_formed("neutral\n\nThe premise says so.")  # label then explanation
    assert not is_well_formed("neutral.")  # trailing punctuation is not one label word
    assert not is_well_formed("entailed, neutral")
    assert not is_well_formed("neutral\n\nWait, the earlier analysis said entailment.")
    assert not is_well_formed("")


def test_format_tracker_writes_snapshots_and_alerts(tmp_path, capsys):
    from anli_classification.rl_train import FormatTracker

    path = tmp_path / "metrics.jsonl"
    tracker = FormatTracker(path, window=4, alert_rate=0.25)
    for well in [True, True, False, True]:  # 1 of 4 malformed: 25%, not above the threshold
        tracker.record(well)
    assert "ALERT" not in capsys.readouterr().out
    for well in [True, False, True]:  # 1 of 3 malformed in the partial window: above the threshold
        tracker.record(well)
    tracker.flush()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["type"] == "format" and "step" not in records[0]
    assert records[0]["format/malformed_rate"] == 0.25
    assert records[1]["window_rollouts"] == 3
    assert "ALERT" in capsys.readouterr().out


def test_dev_strict_accuracy_counts_malformed_answers_as_wrong():
    rows = [{"key": "a"}, {"key": "b"}, {"key": "c"}]
    runs_by_row = [[run(1.0)], [run(1.0)], [run(0.0)]]
    formats = {"a": "parsed", "b": "not_one_label", "c": "unclosed_think"}
    metrics = dev_accuracy(rows, runs_by_row, formats=formats)
    assert metrics["dev/accuracy"] == 2 / 3
    assert metrics["dev/strict_accuracy"] == 1 / 3
    assert metrics["dev/invalid_ratio"] == 2 / 3
    assert formats == {}


def test_epochs_for_steps_leaves_a_margin_for_rejected_groups():
    from anli_classification.rl_train import epochs_for_steps

    # 6 steps x 64 groups = 384 accepted; with 3x margin over 1493 rows that is 1 epoch
    assert epochs_for_steps(6, 64, 1493, 1) == 1
    # a small pool needs more passes: 6 x 64 x 3 / 100 = 11.52 -> 12
    assert epochs_for_steps(6, 64, 100, 1) == 12


def test_max_steps_must_match_checkpoint_and_eval_intervals(capsys):
    import pytest as _pytest

    from anli_classification.rl_train import parse_args

    with _pytest.raises(SystemExit):
        parse_args(["--pool-path", "x", "--max-steps", "5", "--checkpoint-interval", "2", "--eval-interval", "2"])
    args = parse_args(["--pool-path", "x", "--max-steps", "6", "--checkpoint-interval", "2", "--eval-interval", "2"])
    assert args.max_steps == 6


def test_intermediate_checkpoint_names():
    from anli_classification.rl_train import is_intermediate_checkpoint

    assert is_intermediate_checkpoint("step-2")
    assert is_intermediate_checkpoint("step-12")
    assert not is_intermediate_checkpoint("step-2-8377bd45")  # server names, not caller names
    assert not is_intermediate_checkpoint("final-merged")


def test_unparsed_reasons_are_distinct():
    from anli_classification.parsers import unparsed_reason

    assert unparsed_reason("neutral") is None
    assert unparsed_reason("") == "empty"
    assert unparsed_reason("neutral because the premise is silent") == "not_one_label"  # fullmatch, not search
    assert unparsed_reason("<think>x</think>") == "empty_answer"
    assert unparsed_reason("<think>x") == "unclosed_think"
    assert unparsed_reason("<think>a <think>b</think>neutral") == "unbalanced_think"
    assert unparsed_reason("<think>a</think>neutral<think>partial") == "unclosed_think"
    assert unparsed_reason("<think>a</think>neutral") is None
    assert unparsed_reason("<think>a</think>\n<think>b</think>\nneutral") is None  # several closed blocks


def test_unparsed_answers_count_as_incorrect_not_dropped():
    rows = [{"key": "a"}, {"key": "b"}]
    runs_by_row = [[run(0.0)], [run(1.0)]]  # "a" was unparsed: reward 0
    metrics = dev_accuracy(rows, runs_by_row, formats={"a": "not_one_label", "b": "parsed"})
    assert metrics["dev/accuracy"] == 1 / 2
    assert metrics["dev/dropped_ratio"] == 0.0
    assert metrics["dev/strict_accuracy"] == 1 / 2
    assert metrics["dev/invalid_ratio"] == 1 / 2
