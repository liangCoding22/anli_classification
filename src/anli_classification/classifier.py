from __future__ import annotations

from typing import Optional, Sequence

from anli_classification.llm.base import ChatBackend, ChatRequest, ChatResult
from anli_classification.parsers import LabelParser
from anli_classification.prompts import NliPrompt
from anli_classification.schemas import Example, Prediction


def prediction_from_result(
    example: Example, result: Optional[ChatResult], parser: LabelParser
) -> Prediction:
    """Turn one backend reply (or its absence) into a scored Prediction."""
    if result is None:
        return Prediction(example.key, example.label, None, "", "no result returned")
    if result.error:
        return Prediction(
            example.key, example.label, None, result.text, result.error, result.reasoning
        )
    if not result.text.strip():
        # Reasoning models can exhaust max_tokens and return no content.
        error = "empty reply (raise max_tokens)"
        return Prediction(example.key, example.label, None, result.text, error, result.reasoning)
    return Prediction(
        key=example.key,
        gold=example.label,
        pred=parser.parse(result.text),
        raw=result.text,
        reasoning=result.reasoning,
    )


class NliClassifier:
    """Prompt -> LLM backend -> label parser. Knows nothing about datasets or metrics."""

    def __init__(
        self,
        prompt: NliPrompt,
        backend: ChatBackend,
        parser: LabelParser,
        max_tokens: int,
    ) -> None:
        self.prompt = prompt
        self.backend = backend
        self.parser = parser
        self.max_tokens = max_tokens

    def classify_batch(self, examples: Sequence[Example]) -> list[Prediction]:
        if not examples:
            return []
        requests = [
            ChatRequest(
                key=example.key,
                messages=self.prompt.build(example),
                max_tokens=self.max_tokens,
            )
            for example in examples
        ]
        results = {result.key: result for result in self.backend.complete(requests)}
        return [prediction_from_result(ex, results.get(ex.key), self.parser) for ex in examples]
