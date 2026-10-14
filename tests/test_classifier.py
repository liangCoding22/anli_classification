from anli_classification.classifier import NliClassifier
from anli_classification.llm.base import ChatResult
from anli_classification.parsers import LabelParser
from anli_classification.prompts import NliPrompt
from tests.conftest import FakeBackend


def make(reply):
    backend = FakeBackend(reply)
    return NliClassifier(NliPrompt(), backend, LabelParser(), max_tokens=64), backend


def test_classify_batch_one_call_and_parsed_labels(examples):
    classifier, backend = make(lambda r: ChatResult(r.key, "neutral"))
    preds = classifier.classify_batch([(e.key, e) for e in examples])
    assert len(backend.calls) == 1 and len(backend.calls[0]) == len(examples)
    assert [p.key for p in preds] == [e.key for e in examples]
    assert all(p.pred == 1 and p.ok for p in preds)


def test_backend_error_is_failed_and_empty_reply_is_a_wrong_answer(examples):
    def reply(request):
        if request.key == examples[0].key:
            return ChatResult(request.key, "", "RateLimitError: slow down")
        if request.key == examples[1].key:
            return ChatResult(request.key, "   ")
        return None  # backend silently dropped this request

    classifier, _ = make(reply)
    preds = classifier.classify_batch([(e.key, e) for e in examples[:3]])
    assert preds[0].error == "RateLimitError: slow down" and preds[0].pred is None
    assert preds[1].ok and preds[1].pred is None  # scored as wrong, never retried
    assert preds[2].error == "no result returned"


def test_empty_batch_makes_no_backend_call():
    classifier, backend = make(lambda r: None)
    assert classifier.classify_batch([]) == []
    assert backend.calls == []


def test_reasoning_is_kept_on_success_and_on_empty_reply(examples):
    def reply(request):
        if request.key == examples[0].key:
            return ChatResult(request.key, "neutral", reasoning="thinking...")
        return ChatResult(request.key, "", reasoning="ran out of budget")

    classifier, _ = make(reply)
    ok, empty = classifier.classify_batch([(e.key, e) for e in examples[:2]])
    assert ok.reasoning == "thinking..." and ok.pred == 1
    assert empty.ok and empty.pred is None and empty.reasoning == "ran out of budget"
