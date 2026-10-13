import json

from anli_classification.classifier import NliClassifier
from anli_classification.evaluator import Evaluator
from anli_classification.llm.base import ChatResult
from anli_classification.metrics import MetricsCalculator
from anli_classification.parsers import LabelParser
from anli_classification.prompts import NliPrompt
from anli_classification.store import PredictionStore
from tests.conftest import FakeBackend, make_examples


def evaluator_for(tmp_path, reply, batch_size=4):
    backend = FakeBackend(reply)
    classifier = NliClassifier(NliPrompt(), backend, LabelParser(), max_tokens=64)
    return Evaluator(classifier, MetricsCalculator(), tmp_path, batch_size), backend


def gold_label(request):
    # The fake "model" answers correctly by reading the gold label from the uid.
    index = int(request.messages[1]["content"].split("premise ")[1].split("\n")[0])
    return ["entailment", "neutral", "contradiction"][index % 3]


def test_batches_by_batch_size_and_scores_slices(tmp_path):
    examples = make_examples(10)
    evaluator, backend = evaluator_for(
        tmp_path, lambda r: ChatResult(r.key, gold_label(r)), batch_size=4
    )
    report = evaluator.evaluate({"dev_r1": examples})
    assert [len(c) for c in backend.calls] == [4, 4, 2]
    assert report["dev_r1"]["accuracy"] == 1.0
    assert report["overall"]["n"] == 10
    assert json.loads((tmp_path / "metrics.json").read_text())["dev_r1"]["n"] == 10


def test_resume_makes_no_calls_for_cached_predictions(tmp_path):
    examples = make_examples(5)
    evaluator, backend = evaluator_for(tmp_path, lambda r: ChatResult(r.key, "neutral"))
    evaluator.evaluate({"dev_r1": examples})
    first_calls = len(backend.calls)

    evaluator2, backend2 = evaluator_for(tmp_path, lambda r: ChatResult(r.key, "neutral"))
    evaluator2.evaluate({"dev_r1": examples})
    assert backend2.calls == [] and first_calls > 0


def test_errored_predictions_are_retried_on_resume(tmp_path):
    examples = make_examples(3)
    evaluator, _ = evaluator_for(tmp_path, lambda r: ChatResult(r.key, "", "timeout"))
    evaluator.evaluate({"dev_r1": examples})

    evaluator2, backend2 = evaluator_for(tmp_path, lambda r: ChatResult(r.key, "neutral"))
    report = evaluator2.evaluate({"dev_r1": examples})
    assert sum(len(c) for c in backend2.calls) == 3
    assert report["dev_r1"]["errors"] == 0 and report["dev_r1"]["n"] == 3


def test_store_last_line_wins(tmp_path):
    store = PredictionStore(tmp_path / "s.jsonl")
    from anli_classification.schemas import Prediction

    store.append([Prediction("k", 0, None, "", "timeout")])
    store.append([Prediction("k", 0, 0, "entailment")])
    assert store.load()["k"].ok and store.load()["k"].pred == 0


def test_old_cache_lines_without_reasoning_still_load(tmp_path):
    from anli_classification.schemas import Prediction

    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps({"key": "k", "gold": 0, "pred": 0, "raw": "entailment", "error": None}) + "\n")
    pred = PredictionStore(path).load()["k"]
    assert isinstance(pred, Prediction) and pred.reasoning == ""
