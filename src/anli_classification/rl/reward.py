"""The reward of a training rollout: label reward, LLM consistency judge, shaped reward, advantages."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from collections import Counter
from typing import Any
from typing import Optional
from typing import Sequence

from anli_classification.parsers import LabelParser
from anli_classification.rl.defaults import (
    ADVANTAGE_STD_FLOOR,
    JUDGE_MAX_CHARS,
    JUDGE_MODEL,
    MALFORMED_MARGIN,
    NO_REASONING,
)


def label_reward(completion: str, gold: int) -> float:
    """1.0 if the completion's final answer is exactly one label word and it is the gold label, else 0.0."""
    return 1.0 if LabelParser().parse(completion) == gold else 0.0


JUDGE_SYSTEM = (
    "You check a model's reasoning about a natural language inference problem. You are given the premise, the "
    "hypothesis, the model's full reasoning, and the label the model finally output. Decide whether the reasoning "
    "actually arrives at that label. Reply with one line: '<verdict> <label>' where verdict is consistent (the "
    "reasoning concludes the output label), inconsistent (the reasoning concludes a different label), or unclear "
    "(it never settles on one label), and label is the label the reasoning itself concludes (entailment, neutral, "
    "contradiction) or none. Do not judge whether the label is correct."
)


_JUDGE_VERDICT = re.compile(r"^\W*(inconsistent|consistent|unclear)\b", re.IGNORECASE)


def judge_prompt(premise: str, hypothesis: str, thinking: str, label: str) -> str:
    if len(thinking) > JUDGE_MAX_CHARS:
        thinking = thinking[:3000] + "\n[...]\n" + thinking[-12000:]
    return (f"Premise:\n{premise}\n\nHypothesis:\n{hypothesis}\n\nModel's reasoning:\n{thinking}\n\n"
            f"Label the model output: {label}")


def parse_judge_reply(text: Optional[str]) -> Optional[str]:
    """'consistent', 'inconsistent' or 'unclear' from the judge's first word; None when it says anything else.

    Only the verdict is used. The label the judge adds is logged by nobody: on one test case it contradicted its own verdict.
    """
    match = _JUDGE_VERDICT.match(text or "")
    return match.group(1).lower() if match else None


def floored_advantages(rewards: Sequence[float], std_floor: float = ADVANTAGE_STD_FLOOR, eps: float = 1e-8) -> list[float]:
    """The cookbook's group-normalised advantages, but never dividing by a spread below `std_floor`.

    GRPO divides each reward's distance from the group mean by the group's standard deviation. When a small
    reward term (the consistency penalty) is the only difference inside a group, that division stretches it to
    the size of a full right-versus-wrong difference. The floor keeps such a term in proportion to its size.
    Groups with equal rewards still get zero advantage, and `std_floor` 0 reproduces the cookbook.
    """
    n = len(rewards)
    mean = sum(rewards) / n
    std = (sum((r - mean) ** 2 for r in rewards) / (n - 1)) ** 0.5 if n > 1 else 0.0
    if std < eps:
        return [0.0] * n
    return [(r - mean) / (max(std, std_floor) + eps) for r in rewards]


def shaped_reward(label_reward_value: float, verdict: Optional[str], penalty: float) -> float:
    """The label reward, minus `penalty` when the judge found the reasoning concluding a different label.

    The term differs between samples of one prompt, so it survives the cookbook's per-group z-scoring.
    """
    return label_reward_value - (penalty if verdict == "inconsistent" else 0.0)


def training_reward(
    label_value: float,
    pred: Optional[int],
    verdict: Optional[str],
    penalty: float,
    version: str = "v2",
    malformed_margin: float = MALFORMED_MARGIN,
) -> float:
    """The reward a training rollout gets: the label reward, adjusted for the consistency judge.

    v2 (default, the reward of the v2 run): `label_value - penalty` when the verdict is "inconsistent". Two gaps: a
    malformed answer (no judge call) scores 0, which is more than a wrong answer whose reasoning contradicts its
    label (-penalty), and an answer with no reasoning text skips the judge, so it can never be penalised.

    v3 (with penalty > 0) closes both, keeping the order
    correct > wrong > wrong and inconsistent > malformed:
      - malformed (no parseable label): -(penalty + malformed_margin), below every parseable outcome;
      - parseable but no reasoning text (verdict NO_REASONING): `label_value - penalty`, as if inconsistent.
    With penalty 0 every version is the plain label reward.
    """
    if version == "v3" and penalty > 0:
        if pred is None:
            return -(penalty + malformed_margin)
        if verdict == NO_REASONING:
            return label_value - penalty
    return shaped_reward(label_value, verdict, penalty)


class ConsistencyJudge:
    """Asks an LLM whether a rollout's reasoning arrives at the label it output. Fails open.

    The judge sees the premise, hypothesis, full reasoning and the model's output label, never the gold label.
    Any error, timeout or unparseable reply gives no verdict (so no penalty) and is counted, so a judge outage
    cannot stop or distort training. Replies are cached by input.
    """

    def __init__(self, model: str = JUDGE_MODEL, post: Any = None, report_every: int = 500) -> None:
        self.model = model
        self.post = post or self._post
        self.report_every = report_every
        self.stats: Counter = Counter()
        self._cache: dict[str, Optional[str]] = {}
        self._lock = threading.Lock()

    def _post(self, body: dict) -> dict:
        base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
        request = urllib.request.Request(
            f"{base_url}/inference/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())

    def verdict(self, premise: str, hypothesis: str, thinking: str, label: str) -> Optional[str]:
        prompt = judge_prompt(premise, hypothesis, thinking, label)
        with self._lock:
            if prompt in self._cache:
                self.stats["cache_hits"] += 1
                return self._cache[prompt]
        body = {"model": self.model, "temperature": 0, "max_tokens": 300, "reasoning_effort": "low",
                "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}]}
        result: Optional[str] = None
        for attempt in range(3):
            try:
                reply = self.post(body)["choices"][0]["message"].get("content")
                result = parse_judge_reply(reply)
                if result is None:
                    self.stats["unparsed"] += 1
                break
            except Exception:
                if attempt == 2:
                    self.stats["errors"] += 1
                else:
                    time.sleep(1.5 * (attempt + 1))
        with self._lock:
            self._cache[prompt] = result
            self.stats["calls"] += 1
            self.stats[result or "none"] += 1
            if self.stats["calls"] % self.report_every == 0:
                print(self.summary(), flush=True)
        return result

    def record(self, name: str) -> None:
        """Count an event that did not need a call (e.g. a rollout with no reasoning to judge)."""
        with self._lock:
            self.stats[name] += 1

    def summary(self) -> str:
        s = self.stats
        extra = f", no reasoning {s[NO_REASONING]}" if s[NO_REASONING] else ""
        return (f"JUDGE: {s['calls']} calls, inconsistent {s['inconsistent']}, unclear {s['unclear']}, "
                f"errors {s['errors']}, unparsed {s['unparsed']}{extra}")


def thinking_text(message: Any) -> str:
    """The reasoning part of a sampled message (empty when the reply has no thinking part)."""
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return ""
    return "".join(part.get("thinking", "") for part in content if part.get("type") == "thinking")
