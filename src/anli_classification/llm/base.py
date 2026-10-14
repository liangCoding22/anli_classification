from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence


@dataclass(frozen=True)
class ChatRequest:
    key: str  # echoed back in the matching ChatResult
    messages: Sequence[dict[str, str]]
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None  # None leaves the server default; 0 disables top-k
    # Set when the prompt was rendered client-side (rendering.py): the chat backend then sends it
    # to /completions instead of `messages`, so the server's chat template is never applied.
    # `messages` stays for the record. The batch API takes only `messages`.
    prompt: Optional[str] = None


@dataclass(frozen=True)
class ChatResult:
    key: str
    text: str = ""
    error: Optional[str] = None
    reasoning: str = ""


class ChatBackend(Protocol):
    """Anything that turns chat requests into replies, one result per request.

    Implementations must not raise for per-request failures; report them in
    `ChatResult.error` so one bad row cannot sink a whole batch.
    """

    def complete(self, requests: Sequence[ChatRequest]) -> list[ChatResult]:
        ...
