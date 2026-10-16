"""Teacher-data helpers moved out of scripts/: prompt selection, trace filtering, request building, trace collection."""

import json

from anli_classification.teacher_data import build_requests, clean, collect, select_prompts, usable


def example(key, gold, group):
    return {"key": key, "gold": gold, "premise": "p", "hypothesis": "h", "reason": "", "group": group, "n_correct_4b": 0}


def test_selection_keeps_every_mixed_prompt_caps_always_wrong_ones_and_is_deterministic():
    examples = ([example(f"m{g}{i}", g, "1-3/4") for g in range(3) for i in range(4)]
                + [example(f"w{g}{i}", g, "0/4") for g in range(3) for i in range(20)]
                + [example(f"r{g}{i}", g, "4/4") for g in range(3) for i in range(20)])
    chosen = select_prompts(examples, total=60, wrong_cap=15, seed=0)
    keys = [e["key"] for e in chosen]
    assert len(chosen) == 60 and len(set(keys)) == 60
    assert all(f"m{g}{i}" in keys for g in range(3) for i in range(4))  # every mixed prompt
    assert sum(e["group"] == "0/4" for e in chosen) <= 15
    assert keys == [e["key"] for e in select_prompts(examples, total=60, wrong_cap=15, seed=0)]


def trace(**overrides):
    base = {"correct": True, "finish_reason": "stop", "completion_tokens": 500, "reasoning": "It follows. Wait, no. Yes."}
    return {**base, **overrides}


def test_a_trace_is_usable_only_when_correct_complete_short_and_not_dithering():
    assert usable(trace(), max_tokens=3000, max_waits=3)
    assert not usable(trace(correct=False), 3000, 3)
    assert not usable(trace(finish_reason="length"), 3000, 3)
    assert not usable(trace(completion_tokens=4000), 3000, 3)
    assert not usable(trace(reasoning="Wait. " * 5), 3000, 3)
    assert not usable(trace(reasoning="   "), 3000, 3)
    assert not usable({**trace(), "error": "boom"}, 3000, 3)


def test_optional_filters_drop_chinese_text_and_claims_about_nli_conventions():
    chinese = trace(reasoning="so the answer is neutral 因此")
    claim = trace(reasoning="Under the standard NLI protocol this is neutral.")
    assert usable(chinese, 3000, 3) and not usable(chinese, 3000, 3, drop_cjk=True)
    assert usable(claim, 3000, 3) and not usable(claim, 3000, 3, drop_protocol_claims=True)


def test_clean_strips_a_stray_trailing_marker():
    assert clean("The premise says so. .cw  ") == "The premise says so. ."
    assert clean("  plain  ") == "plain"


def test_requests_are_keyed_per_sample_and_honour_a_row_level_sample_count():
    rows = [{"key": "a", "premise": "p", "hypothesis": "h", "gold": 1},
            {"key": "b", "premise": "p", "hypothesis": "h", "gold": 0, "samples": 3}]
    requests = build_requests(rows, default_samples=2, max_tokens=100, temperature=0.6, top_p=0.95)
    assert [r.key for r in requests] == ["a#0", "a#1", "b#0", "b#1", "b#2"]
    assert all(r.max_tokens == 100 and r.temperature == 0.6 and r.top_p == 0.95 for r in requests)


def test_collect_turns_batch_output_into_traces_with_correctness_and_error_rows(tmp_path):
    chunk = tmp_path / "chunk_00"
    chunk.mkdir()
    ok = {"custom_id": "a#0", "response": {"choices": [{"message": {"content": "neutral", "reasoning_content": " thinking "},
                                                          "finish_reason": "stop"}], "usage": {"completion_tokens": 42}}}
    wrong = {"custom_id": "b#1", "response": {"choices": [{"message": {"content": "entailment", "reasoning_content": "r"},
                                                             "finish_reason": "stop"}], "usage": {"completion_tokens": 7}}}
    failed = {"custom_id": "c#0", "error": {"code": 500}}
    (chunk / "output.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (ok, wrong, failed)))
    assert collect(tmp_path, {"a": 1, "b": 2, "c": 0}) == 3
    traces = [json.loads(line) for line in (tmp_path / "traces.jsonl").read_text().splitlines()]
    a, b, c = traces
    assert (a["pred"], a["correct"], a["reasoning"], a["completion_tokens"]) == ("neutral", True, "thinking", 42)
    assert (b["pred"], b["correct"]) == ("entailment", False)
    assert "error" in c and "pred" not in c
