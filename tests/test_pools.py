"""Teacher-verified pools: the selection rules, determinism, and an exact rebuild of the committed pool_v2."""

import json
from pathlib import Path

import pytest

from anli_classification.pools import build_clean_pools, find_hard_prompts, load_votes, teacher_verdict

REPRO = Path(__file__).resolve().parent.parent / "reproducibility" / "pools"
FIELDS = ("key", "gold", "premise", "hypothesis", "reason", "group", "n_correct", "n_samples", "teacher_votes")
L = ("entailment", "neutral", "contradiction")


def row(key, gold, n_correct=2):
    return {"key": key, "gold": gold, "premise": f"p {key}", "hypothesis": f"h {key}", "reason": "", "group": "1-3/4",
            "n_correct": n_correct, "n_samples": 4}


def synthetic(n_entail=6, n_neutral=12, n_contra=9, disagree=3):
    rows, votes = [], {}
    for gold, n in ((0, n_entail), (1, n_neutral), (2, n_contra)):
        for i in range(n):
            key = f"{L[gold][:3]}{i}"
            rows.append(row(key, gold))
            votes[key] = [L[gold], L[gold]]
    for i in range(disagree):  # the teacher rejects gold on these neutral prompts
        key = f"bad{i}"
        rows.append(row(key, 1))
        votes[key] = [L[2], L[2]]
    return rows, votes


def test_teacher_verdict_agree_split_disagree_and_uncovered():
    assert teacher_verdict(["neutral", "neutral"], "neutral") == "agree"
    assert teacher_verdict(["neutral", "entailment"], "neutral") == "split"
    assert teacher_verdict(["entailment", "entailment"], "neutral") == "disagree"
    assert teacher_verdict([], "neutral") == "uncovered"


def test_load_votes_skips_unparsed_answers_and_reads_files_in_sorted_order(tmp_path):
    def write(name, traces):
        (tmp_path / name).write_text("".join(json.dumps(t) + "\n" for t in traces))

    write("b.jsonl", [{"key": "k", "pred": "neutral", "unparsed_reason": None}])
    write("a.jsonl", [{"key": "k", "pred": "entailment", "unparsed_reason": None},
                      {"key": "k", "pred": None, "unparsed_reason": "empty"},
                      {"key": "k", "pred": "neutral", "unparsed_reason": "not_one_label"}])
    assert load_votes(str(tmp_path / "*.jsonl")) == {"k": ["entailment", "neutral"]}  # a.jsonl is read before b.jsonl


def test_only_prompts_whose_every_vote_agrees_are_kept_and_classes_are_balanced():
    rows, votes = synthetic()
    pools = build_clean_pools(rows, votes, seed=0)
    assert pools.per_class == 6  # the smallest class after filtering
    assert all(r["teacher_votes"] == [L[r["gold"]]] * 2 for r in pools.balanced)
    assert not any(r["key"].startswith("bad") for r in pools.balanced + pools.unbalanced)
    counts = [sum(r["gold"] == g for r in pools.balanced) for g in range(3)]
    assert counts == [6, 6, 6] and len(pools.unbalanced) == 6 + 12 + 9
    assert pools.verdicts[("neutral", "disagree")] == 3


def test_the_same_seed_gives_the_same_pool_and_another_seed_the_same_rows_in_another_order():
    rows, votes = synthetic()
    first = build_clean_pools(rows, votes, seed=0, neutral_share=0.4)
    again = build_clean_pools(rows, votes, seed=0, neutral_share=0.4)
    other = build_clean_pools(rows, votes, seed=1, neutral_share=0.4)
    assert [r["key"] for r in first.v2] == [r["key"] for r in again.v2]
    assert [r["key"] for r in first.balanced] == [r["key"] for r in again.balanced]
    assert [r["key"] for r in first.v2] != [r["key"] for r in other.v2]


def test_neutral_share_sets_the_neutral_fraction_without_duplicate_prompts():
    rows, votes = synthetic(n_entail=6, n_neutral=40, n_contra=12)
    pools = build_clean_pools(rows, votes, seed=0, neutral_share=0.36)
    keys = [r["key"] for r in pools.v2]
    assert len(keys) == len(set(keys))
    assert abs(pools.v2_summary["neutral_share"] - 0.36) < 0.05
    # the share cannot exceed what the teacher verified: with few neutral prompts it is capped
    capped = build_clean_pools(*synthetic(n_entail=6, n_neutral=4, n_contra=6), seed=0, neutral_share=0.6)
    assert capped.v2_summary["by_label"]["neutral"] == 4


def test_hard_prompts_need_a_0_of_4_a_new_key_and_every_teacher_vote_agreeing():
    votes = {"h1": ["entailment", "entailment"], "h2": ["neutral", "entailment"], "h3": ["entailment"], "h4": ["entailment"]}
    prompts = [
        {"key": "h1", "gold": 0, "n_correct_4b": 0}, {"key": "h1", "gold": 0, "n_correct_4b": 0},  # first occurrence wins
        {"key": "h2", "gold": 0, "n_correct_4b": 0},  # a split teacher
        {"key": "h3", "gold": 0, "n_correct_4b": 2},  # the 4B did not fail
        {"key": "h4", "gold": 0, "n_correct_4b": 0},  # already in the pool
    ]
    hard = find_hard_prompts(prompts, {"h4"}, votes)
    assert list(hard) == ["h1"] and hard["h1"]["n_correct"] == 0 and hard["h1"]["teacher_votes"] == votes["h1"]


def test_hard_prompts_enter_pool_v2_and_contradiction_is_topped_up_to_match_entailment():
    rows, votes = synthetic(n_entail=5, n_neutral=20, n_contra=30)
    hard = {f"hard{i}": {**row(f"hard{i}", 0, 0), "teacher_votes": ["entailment"] * 2} for i in range(4)}
    pools = build_clean_pools(rows, votes, seed=0, neutral_share=0.3, hard=hard)
    by = pools.v2_summary["by_label"]
    assert pools.v2_summary["hard_rows"] == 4 and by["entailment"] == 5 + 4
    assert by["contradiction"] == by["entailment"]


def test_the_committed_pool_v2_rebuilds_exactly_from_the_committed_files():
    needed = ("pool_mixed.compact.jsonl", "teacher_votes.jsonl", "hard_candidates.compact.jsonl", "pool_v2.compact.jsonl",
              "pool_clean.compact.jsonl")
    if not all((REPRO / name).exists() for name in needed):
        pytest.skip("the committed pool files are not present")

    def read(name):
        return [json.loads(line) for line in (REPRO / name).read_text().splitlines() if line.strip()]

    votes = {r["key"]: r["votes"] for r in read("teacher_votes.jsonl")}
    hard = {r["key"]: r for r in read("hard_candidates.compact.jsonl")}  # in the order pool_v2 was built with
    pools = build_clean_pools(read("pool_mixed.compact.jsonl"), votes, seed=0, neutral_share=0.36, hard=hard)

    def project(rows):
        return [{f: r.get(f) for f in FIELDS} for r in rows]

    assert project(pools.v2) == project(read("pool_v2.compact.jsonl"))
    assert project(pools.balanced) == project(read("pool_clean.compact.jsonl"))
