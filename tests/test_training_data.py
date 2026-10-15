from anli_classification.schemas import Prediction, sample_key
from anli_classification.training_data import categorize, group_by_example, select_traces


def pred(key, gold, pred_label, error=None, reasoning="think"):
    return Prediction(key=key, gold=gold, pred=pred_label, raw=f"raw{pred_label}", error=error, reasoning=reasoning)


def example(key, gold, answers, error_at=None):
    """answers: predicted labels for samples 0..n-1."""
    out = {}
    for i, label in enumerate(answers):
        out[i] = pred(sample_key(key, i), gold, label, error="boom" if i == error_at else None)
    return out


def test_categorize_by_correct_count():
    assert categorize(example("a", 0, [0, 0, 0, 0])) == "all_correct"
    assert categorize(example("a", 0, [1, 1, 1, 1])) == "all_wrong"
    assert categorize(example("a", 0, [0, 1, 1, 1])) == "mixed"
    assert categorize(example("a", 0, [0] * 8)) == "all_correct"
    assert categorize(example("a", 0, [0, 1, 1, 1, 1, 1, 1, 0])) == "mixed"  # 2 of 8
    assert categorize(example("a", 0, [0, 0, 0, 0], error_at=2)) == "incomplete"


def test_group_by_example_separates_sample_indices():
    preds = {}
    for key, samples in {"x": example("x", 0, [0, 1]), "y": example("y", 1, [1])}.items():
        preds.update({p.key: p for p in samples.values()})
    grouped = group_by_example(preds)
    assert sorted(grouped) == ["x", "y"]
    assert sorted(grouped["x"]) == [0, 1]


def test_all_mixed_kept_one_correct_trace_each():
    examples = {
        "m1": example("m1", 0, [0, 1, 0, 1]),
        "m2": example("m2", 2, [2, 2, 0, 1, 1, 1, 2, 2]),
        "w": example("w", 1, [0, 0, 0, 0]),  # all wrong: dropped
        "i": example("i", 1, [1, 1, 1, 1], error_at=3),  # incomplete: dropped
    }
    sources = {k: "run" for k in examples}
    records, counts = select_traces(examples, sources, all_correct_rate=0.0, seed=0)
    assert sorted(r["key"] for r in records) == ["m1", "m2"]
    for r in records:
        assert r["category"] == "mixed"
        assert r["trace"]["pred"] == r["label"]  # the kept trace is a correct one
        assert r["trace"]["reasoning"] == "think"
        assert r["trace"]["raw"].startswith("raw")
    assert counts["all_wrong"] == 1 and counts["incomplete"] == 1


def test_all_correct_rate_drawn_across_examples():
    examples = {f"c{i}": example(f"c{i}", 0, [0, 0, 0, 0]) for i in range(100)}
    sources = {k: "run" for k in examples}
    records, counts = select_traces(examples, sources, all_correct_rate=0.05, seed=0)
    assert len(records) == 5
    assert counts["all_correct_kept"] == 5
    assert len({r["key"] for r in records}) == 5  # one trace per example
    assert all(r["category"] == "all_correct" for r in records)


def test_selection_is_reproducible_for_a_seed():
    examples = {f"c{i}": example(f"c{i}", 0, [0, 0, 0, 0]) for i in range(50)}
    examples.update({f"m{i}": example(f"m{i}", 0, [0, 1, 1, 0]) for i in range(10)})
    sources = {k: "run" for k in examples}
    first, _ = select_traces(examples, sources, all_correct_rate=0.2, seed=3)
    second, _ = select_traces(examples, sources, all_correct_rate=0.2, seed=3)
    assert first == second
