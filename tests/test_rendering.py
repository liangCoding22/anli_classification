import json

import pytest

from anli_classification.classifier import NliClassifier
from anli_classification.config import run_dir_for
from anli_classification.llm.base import ChatResult
from anli_classification.llm.fireworks import parse_batch_results
from anli_classification.metrics import MetricsCalculator, no_reasoning_warning
from anli_classification.parsers import LabelParser, split_reasoning
from anli_classification.prompts import NliPromptV3
from anli_classification.rl_train import label_reward
from anli_classification.schemas import Prediction
from tests.conftest import FakeBackend


class FakeRenderer:
    def render(self, messages):
        return "|".join(m["content"] for m in messages) + "|assistant"


def test_split_reasoning_moves_one_leading_think_block_out():
    assert split_reasoning("<think>\nhmm\n</think>\n\nneutral") == ("\nhmm\n", "\n\nneutral")
    assert split_reasoning("neutral") == ("", "neutral")


def test_split_reasoning_leaves_malformed_replies_unchanged():
    for text in ("<think>never closes entailment", "a</think>b</think>neutral", "<think>a</think>x<think>b</think>neutral"):
        assert split_reasoning(text) == ("", text)


def test_reward_scores_only_the_final_answer_not_the_thinking():
    # Text as the RL rollout grades it: the renderer already moved the thinking to its own part.
    assert label_reward("\n\nneutral", 1) == 1.0
    # A label inside the thinking never counts, even when the answer is missing or wrong.
    assert label_reward("<think>neutral</think>\n\n", 1) == 0.0
    assert label_reward("<think>neutral</think>\n\nentailment", 1) == 0.0
    assert label_reward("<think>truncated, neutral", 1) == 0.0


def test_classifier_with_renderer_sends_rendered_prompt(examples):
    seen = []

    def reply(request):
        seen.append(request)
        return ChatResult(request.key, "entailment", reasoning="r")

    clf = NliClassifier(NliPromptV3(), FakeBackend(reply), LabelParser(), max_tokens=8, renderer=FakeRenderer())
    clf.classify_batch([(examples[0].key, examples[0])])
    request = seen[0]
    assert request.prompt.endswith("|assistant")
    assert request.messages  # kept for the record


def test_client_template_is_refused_for_the_batch_backend():
    from anli_classification.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--template", "client", "--backend", "batch", "--account-id", "a"])
    assert parse_args(["--template", "client"]).template == "client"


def test_batch_results_with_inline_thinking_are_split():
    row = {"custom_id": "a", "response": {"choices": [{"message": {"content": "<think>x</think>\n\nneutral"}}]}}
    assert parse_batch_results(json.dumps(row))["a"] == ChatResult("a", "\n\nneutral", reasoning="x")


def test_client_template_runs_get_their_own_directory(tmp_path):
    server = run_dir_for(tmp_path, "accounts/a/models/m", "zero-shot-v3")
    client = run_dir_for(tmp_path, "accounts/a/models/m", "zero-shot-v3", "client")
    assert server.name == "zero-shot-v3" and client.name == "zero-shot-v3-client-template"


def test_no_reasoning_warning_only_when_no_reply_reasoned():
    def metrics(reasonings):
        return MetricsCalculator().compute(
            [Prediction(key=f"k{i}", gold=0, pred=0, raw="entailment", reasoning=r) for i, r in enumerate(reasonings)]
        )

    assert no_reasoning_warning("dev_r3", metrics(["", ""])) is not None
    assert no_reasoning_warning("dev_r3", metrics(["", "thought"])) is None
    assert no_reasoning_warning("dev_r3", metrics([])) is None


def test_hf_qwen3_template_matches_training_prompt():
    """With thinking on, the prompt ends at the assistant header: no empty <think></think> prefill."""
    pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    from anli_classification.rendering import ChatTemplateRenderer

    try:
        prompt = ChatTemplateRenderer("Qwen/Qwen3-4B").render([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    except OSError:
        pytest.skip("Qwen/Qwen3-4B tokenizer not available offline")
    assert prompt == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
