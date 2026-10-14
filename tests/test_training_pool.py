from __future__ import annotations

import json

import pytest

from anli_classification.classifier import NliClassifier
from anli_classification.llm.base import ChatResult
from anli_classification.parsers import LabelParser
from anli_classification.prompts import NliPrompt
from anli_classification.schemas import LABELS, sample_key
from anli_classification.store import PredictionStore
from anli_classification.training_pool import (
    GROUP_ALL_CORRECT,
    GROUP_ALL_WRONG,
    GROUP_MIXED,
    build_training_pool,
)
from tests.conftest import FakeBackend, make_examples


def target_correct(index: int) -> int:
    """Example i is answered correctly on the first `target` of 8 samples, target = i % 9."""
    return index % 9


def scripted_reply(request):
    """Correct on sample s iff s < target; the uid carries the example index."""
    uid, _, sample = request.key.split("/", 1)[1].partition("#")
    index = int(uid[1:])
    gold = index % 3
    correct = int(sample or 0) < target_correct(index)
    return ChatResult(request.key, LABELS[gold] if correct else LABELS[(gold + 1) % 3])


def pool_for(backend, examples, out_dir, edge_rate=0.2):
    classifier = NliClassifier(NliPrompt(), backend, LabelParser(), max_tokens=64)
    return build_training_pool(
        classifier, examples, out_dir, batch_size=1000, edge_rate=edge_rate, seed=0, settings={}
    )


def test_selects_mixed_and_edge_groups_and_builds_eight_answers(tmp_path):
    examples = make_examples(20)
    summary = pool_for(FakeBackend(scripted_reply), examples, tmp_path)

    targets = {ex.key: target_correct(i) for i, ex in enumerate(examples)}
    mixed = [k for k, t in targets.items() if 1 <= t <= 3]
    assert summary["selection"]["available"][GROUP_MIXED] == len(mixed)
    assert summary["selection"]["edge_quota"] == 4
    assert summary["selection"]["selected"][GROUP_MIXED] == len(mixed)
    assert summary["selection"]["selected"][GROUP_ALL_CORRECT] == 4
    assert summary["selection"]["selected"][GROUP_ALL_WRONG] == 3  # only 3 examples have target 0

    records = [json.loads(line) for line in (tmp_path / "pool.jsonl").read_text().splitlines()]
    assert len(records) == summary["pool_size"]
    for record in records:
        assert record["n_correct"] == sum(s["correct"] for s in record["samples"])
        assert record["n_correct"] == targets[record["key"]]
        if record["group"] == GROUP_MIXED:
            assert record["n_samples"] == len(record["samples"]) == 4  # no second round
        else:
            assert record["n_samples"] == len(record["samples"]) == 8
    # mixed examples are all in the pool; edge examples only if round 2 shows them mixed
    assert summary["pool_from_mixed"] == len(mixed)
    edge_records = [r for r in records if r["group"] != GROUP_MIXED]
    edge_selected = summary["selection"]["selected"][GROUP_ALL_CORRECT] + summary["selection"]["selected"][GROUP_ALL_WRONG]
    assert sum(summary["distribution"][GROUP_ALL_CORRECT].values()) + sum(
        summary["distribution"][GROUP_ALL_WRONG].values()
    ) == edge_selected
    assert summary["pool_from_edge"] == len(edge_records) == sum(0 < r["n_correct"] < 8 for r in edge_records)


def test_distribution_by_group_uses_four_or_eight_answers(tmp_path):
    examples = make_examples(20)
    summary = pool_for(FakeBackend(scripted_reply), examples, tmp_path)
    dist = summary["distribution"]
    assert sorted(dist[GROUP_MIXED]) == list(range(5))  # 0-4 of 4
    assert sorted(dist[GROUP_ALL_CORRECT]) == list(range(9))  # 0-8 of 8
    pool = summary["pool_distribution"]
    assert sum(pool[GROUP_MIXED].values()) + sum(pool[GROUP_ALL_CORRECT].values()) + sum(
        pool[GROUP_ALL_WRONG].values()
    ) == summary["pool_size"]
    # an edge example that stays all-right or all-wrong over 8 answers is kept out of the pool
    assert pool[GROUP_ALL_CORRECT][8] == pool[GROUP_ALL_WRONG][0] == 0


def test_round_one_answers_samples_0_to_3_and_round_two_adds_4_to_7(tmp_path):
    examples = make_examples(10)
    backend = FakeBackend(scripted_reply)
    summary = pool_for(backend, examples, tmp_path)

    keys = [key for call in backend.calls for key in call]
    round_one = [k for k in keys if k.partition("#")[2] in ("", "1", "2", "3")]
    round_two = [k for k in keys if k.partition("#")[2] in ("4", "5", "6", "7")]
    assert len(round_one) == len(set(round_one)) == 10 * 4
    edge = summary["selection"]["selected"][GROUP_ALL_CORRECT] + summary["selection"]["selected"][GROUP_ALL_WRONG]
    assert len(round_two) == len(set(round_two)) == edge * 4  # only edge examples get round 2


def test_round_two_is_not_submitted_when_round_one_job_fails(tmp_path):
    examples = make_examples(10)
    backend = FakeBackend(scripted_reply)
    calls = []

    def failing_round_one(requests):
        calls.append(len(requests))
        if len(calls) == 1:
            raise RuntimeError("batch job anli-job-x ended in state FAILED")
        return backend.complete(requests)

    class Backend:
        def complete(self, requests):
            return failing_round_one(requests)

    with pytest.raises(RuntimeError):
        pool_for(Backend(), examples, tmp_path)
    assert calls == [40]  # only the round-1 request set was sent


def test_resume_makes_no_backend_calls(tmp_path):
    examples = make_examples(20)
    pool_for(FakeBackend(scripted_reply), examples, tmp_path)

    rerun = FakeBackend(scripted_reply)
    summary = pool_for(rerun, examples, tmp_path)
    assert rerun.calls == []
    assert summary["pool_size"] > 0


def test_examples_with_a_failed_answer_are_left_out(tmp_path):
    examples = make_examples(10)
    failing = sample_key(examples[0].key, 2)

    def reply(request):
        if request.key == failing:
            return ChatResult(request.key, "", "timeout")
        return scripted_reply(request)

    summary = pool_for(FakeBackend(reply), examples, tmp_path)
    pooled = [json.loads(line)["key"] for line in (tmp_path / "pool.jsonl").read_text().splitlines()]
    assert examples[0].key in summary["round1_incomplete"]
    assert examples[0].key not in pooled
    assert PredictionStore(tmp_path / "predictions.jsonl").load()[failing].error == "timeout"


def test_draw_examples_skips_excluded_and_returns_n():
    from anli_classification.training_pool import draw_examples

    candidates = make_examples(50)
    excluded = {ex.key for ex in candidates[:30]}
    drawn = draw_examples(candidates, 10, seed=1, excluded=excluded)
    assert len(drawn) == 10 and len({ex.key for ex in drawn}) == 10
    assert not {ex.key for ex in drawn} & excluded


def test_draw_examples_rejects_too_large_n():
    from anli_classification.training_pool import draw_examples

    candidates = make_examples(5)
    with pytest.raises(ValueError):
        draw_examples(candidates, 6, seed=0, excluded=set())
