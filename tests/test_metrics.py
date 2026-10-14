import pytest

from anli_classification.metrics import MetricsCalculator
from anli_classification.schemas import Prediction


def pred(gold, p, error=None):
    return Prediction(key=f"k{gold}{p}", gold=gold, pred=p, raw="", error=error)


def test_perfect_predictions():
    m = MetricsCalculator().compute([pred(g, g) for g in (0, 1, 2, 0, 1, 2)])
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["unparsed"] == 0 and m["errors"] == 0


def test_unparsed_and_errors_count_as_wrong_and_skip_confusion():
    m = MetricsCalculator().compute(
        [pred(0, 0), pred(1, None), pred(2, 1, error="boom")]
    )
    assert m["accuracy"] == pytest.approx(1 / 3)
    assert m["unparsed"] == 1 and m["errors"] == 1
    assert sum(map(sum, m["confusion"]["matrix"])) == 2
    assert m["confusion"]["matrix"][2][1] == 1


def test_macro_f1_hand_computed():
    # gold 0 -> pred 0, 1; gold 1 -> pred 1; gold 2 -> pred 2
    m = MetricsCalculator().compute([pred(0, 0), pred(0, 1), pred(1, 1), pred(2, 2)])
    # class 0: P=1, R=0.5, F1=2/3; class 1: P=0.5, R=1, F1=2/3; class 2: F1=1
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_empty_input():
    m = MetricsCalculator().compute([])
    assert m["n"] == 0 and m["accuracy"] == 0.0


def sample(key, gold, p):
    return Prediction(key=key, gold=gold, pred=p, raw="")


def test_pass_at_k_over_three_samples_per_example():
    # A: 1 of 3 correct. B: 3 of 3 correct. C: 0 of 3 correct.
    preds = [
        sample("dev_r1/A", 0, 0), sample("dev_r1/A#1", 0, 1), sample("dev_r1/A#2", 0, 2),
        sample("dev_r1/B", 1, 1), sample("dev_r1/B#1", 1, 1), sample("dev_r1/B#2", 1, 1),
        sample("dev_r1/C", 2, 0), sample("dev_r1/C#1", 2, 0), sample("dev_r1/C#2", 2, 0),
    ]
    m = MetricsCalculator().compute(preds)["pass_at_k"]
    assert m["examples"] == 3
    assert m["pass@1"] == pytest.approx((1 / 3 + 1 + 0) / 3)  # mean per-sample correct rate
    assert m["pass@3"] == pytest.approx(2 / 3)  # any of the three samples correct


def test_pass_at_k_skips_examples_with_too_few_samples():
    preds = [sample("dev_r1/A", 0, 0), sample("dev_r1/B", 1, 0)]
    m = MetricsCalculator().compute(preds)["pass_at_k"]
    assert m["pass@1"] == pytest.approx(0.5)
    assert m["pass@3"] is None


def test_strict_accuracy_counts_malformed_answers_as_wrong():
    from anli_classification.metrics import MetricsCalculator
    from anli_classification.schemas import Prediction

    def pred(gold, pred_label, raw):
        return Prediction(key=f"k{raw}", gold=gold, pred=pred_label, raw=raw)

    predictions = [
        pred(1, 1, "neutral"),  # exact, correct
        pred(0, 0, "entailment\n\nThe premise implies it."),  # label then text: well formed
        pred(2, 2, "contradiction\n\nWait, the earlier analysis said neutral."),  # several labels
        pred(1, 1, "The answer is neutral."),  # label not first
    ]
    m = MetricsCalculator().compute(predictions)
    assert m["accuracy"] == 4 / 4  # first-match scoring, unchanged
    assert m["malformed"] == 3  # label then text, several labels, label not first
    assert m["strict_accuracy"] == 1 / 4  # only the exact reply counts
