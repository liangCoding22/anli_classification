"""The reported numbers must be recomputable from the stored files, and the stored pools must match the manifest."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "reproducibility"
pytestmark = pytest.mark.skipif(not (ROOT / "manifest.json").exists(), reason="reproducibility files not present")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def accuracy(rows):
    return sum(r["pred"] == r["gold"] for r in rows) / len(rows)


def test_reported_test_r3_accuracies_recompute_from_stored_predictions():
    promoted = read_jsonl(ROOT / "results" / "promoted" / "test_r3.compact.jsonl")
    base = read_jsonl(ROOT / "results" / "base" / "test_r3.compact.jsonl")
    assert len(promoted) == len(base) == 1200
    assert len({r["key"] for r in promoted}) == 1200  # a retried key must not appear twice
    assert round(accuracy(promoted), 4) == 0.7008
    assert round(accuracy(base), 4) == 0.6367
    assert sorted(r["key"] for r in promoted) == sorted(r["key"] for r in base)  # paired on the same examples


def test_stored_metrics_json_agrees_with_the_predictions():
    for tag in ("promoted", "base"):
        metrics = json.loads((ROOT / "results" / tag / "metrics.json").read_text())["test_r3"]
        rows = read_jsonl(ROOT / "results" / tag / "test_r3.compact.jsonl")
        assert metrics["n"] == len(rows)
        assert round(metrics["accuracy"], 4) == round(accuracy(rows), 4)


def test_compact_pools_match_the_manifest_and_load_as_training_rows():
    from anli_classification.rl_train import load_pool_rows

    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, info in manifest["pools"].items():
        path = ROOT / info["compact_file"]
        rows = read_jsonl(path)
        assert len(rows) == info["rows"], name
        assert len({r["key"] for r in rows}) == len(rows), name
    v2 = load_pool_rows(ROOT / "pools" / "pool_v2.compact.jsonl")
    labels = [row["gold"] for row in v2]
    assert (labels.count(0), labels.count(1), labels.count(2)) == (1259, 1419, 1259)
    assert all(row["premise"] and row["hypothesis"] for row in v2)


def test_every_pool_prompt_has_teacher_votes_recorded():
    votes = {r["key"]: r for r in read_jsonl(ROOT / "pools" / "teacher_votes.jsonl")}
    pool = read_jsonl(ROOT / "pools" / "pool_mixed.compact.jsonl")
    assert all(r["key"] in votes and votes[r["key"]]["votes"] for r in pool)
