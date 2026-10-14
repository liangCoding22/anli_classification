from __future__ import annotations

from typing import Optional, Sequence

from anli_classification.llm.base import ChatBackend, ChatRequest, ChatResult
from anli_classification.parsers import LabelParser
from anli_classification.prompts import NliPrompt
from anli_classification.schemas import Example, Prediction


def prediction_from_result(
    example: Example, key: str, result: Optional[ChatResult], parser: LabelParser
) -> Prediction:
    """Turn one backend reply (or its absence) into a scored Prediction stored under `key`."""
    source = dict(
        key=key,
        gold=example.label,
        premise=example.premise,
        hypothesis=example.hypothesis,
        reason=example.reason,
    )
    if result is None:
        return Prediction(**source, pred=None, raw="", error="no result returned")
    if result.error:
        return Prediction(
            **source, pred=None, raw=result.text, error=result.error, reasoning=result.reasoning
        )
    if not result.text.strip():
        # Reasoning models can exhaust max_tokens and return no content.
        return Prediction(
            **source,
            pred=None,
            raw=result.text,
            error="empty reply (raise max_tokens)",
            reasoning=result.reasoning,
        )
    return Prediction(
        **source,
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
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
    ) -> None:
        self.prompt = prompt
        self.backend = backend
        self.parser = parser
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def classify_batch(self, items: Sequence[tuple[str, Example]]) -> list[Prediction]:
        """Classify `(prediction_key, example)` pairs; one prediction per pair, in order."""
        if not items:
            return []
        requests = [
            ChatRequest(
                key=key,
                messages=self.prompt.build(example),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            for key, example in items
        ]
        results = {result.key: result for result in self.backend.complete(requests)}
        return [
            prediction_from_result(example, key, results.get(key), self.parser)
            for key, example in items
        ]
