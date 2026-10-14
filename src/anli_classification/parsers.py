from __future__ import annotations

import re
from typing import Optional

from anli_classification.schemas import LABEL2ID

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
_TAG = re.compile(r"</?think>")
_LABEL = re.compile(r"(entailment|neutral|contradiction)", re.IGNORECASE)


def unparsed_reason(text: Optional[str]) -> Optional[str]:
    """Why a reply cannot be scored, or None when its final answer is exactly one label word.

    Reasons: 'empty' (no text), 'unbalanced_think' (a thinking tag is unmatched somewhere),
    'unclosed_think' (a thinking block never closes; its reasoning is not used),
    'empty_answer' (nothing after the thinking), 'not_one_label' (the answer is not exactly one label).

    Replies without thinking tags are their own answer; thinking separated by the API needs no tags.
    """
    if text is None or not text.strip():
        return "empty"
    tags = [m.group(0) for m in _TAG.finditer(text)]
    for i, tag in enumerate(tags):
        if tag != (THINK_OPEN if i % 2 == 0 else THINK_CLOSE):
            return "unbalanced_think"
    if len(tags) % 2:
        return "unclosed_think"
    answer = text.rsplit(THINK_CLOSE, 1)[1] if tags else text
    answer = answer.strip()
    if not answer:
        return "empty_answer"
    if _LABEL.fullmatch(answer) is None:
        return "not_one_label"
    return None


def split_reasoning(text: str) -> tuple[str, str]:
    """(reasoning, answer) for a reply that is one leading `<think>...</think>` block plus an answer.

    Raw completions keep the thinking inline, where the chat API would have moved it to a separate
    field; this gives them the same shape. Any other reply comes back as ('', text) unchanged, so
    `unparsed_reason` still sees the malformed tags.
    """
    tags = [m.group(0) for m in _TAG.finditer(text)]
    if tags != [THINK_OPEN, THINK_CLOSE] or not text.lstrip().startswith(THINK_OPEN):
        return "", text
    reasoning, answer = text.lstrip()[len(THINK_OPEN):].split(THINK_CLOSE, 1)
    return reasoning, answer


def parse_answer(text: Optional[str]) -> Optional[int]:
    """Label id of the final answer when it is exactly one label word (case-insensitive), else None."""
    if unparsed_reason(text) is not None:
        return None
    answer = text.rsplit(THINK_CLOSE, 1)[1] if THINK_CLOSE in text else text
    return LABEL2ID[answer.strip().lower()]


class LabelParser:
    """Maps a model reply to a label id. Used by the eval, the RL reward, and the dev metrics alike."""

    def parse(self, text: Optional[str]) -> Optional[int]:
        return parse_answer(text)


def is_well_formed(text: Optional[str]) -> bool:
    """True when the reply parses to a label, i.e. its final answer is exactly one label word."""
    return unparsed_reason(text) is None
