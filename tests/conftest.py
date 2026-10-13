from __future__ import annotations

from typing import Callable, Optional, Sequence

import pytest

from anli_classification.llm.base import ChatRequest, ChatResult
from anli_classification.schemas import Example


class FakeBackend:
    """Replies from a function of the request; records every call for assertions."""

    def __init__(self, reply: Callable[[ChatRequest], Optional[ChatResult]]) -> None:
        self.reply = reply
        self.calls: list[list[str]] = []

    def complete(self, requests: Sequence[ChatRequest]) -> list[ChatResult]:
        self.calls.append([r.key for r in requests])
        out = []
        for request in requests:
            result = self.reply(request)
            if result is not None:
                out.append(result)
        return out


def make_examples(n: int, label_of: Callable[[int], int] = lambda i: i % 3) -> list[Example]:
    return [
        Example(
            split="dev",
            round_id="r1",
            uid=f"u{i}",
            premise=f"premise {i}",
            hypothesis=f"hypothesis {i}",
            label=label_of(i),
        )
        for i in range(n)
    ]


@pytest.fixture
def examples() -> list[Example]:
    return make_examples(6)
