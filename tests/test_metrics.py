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
