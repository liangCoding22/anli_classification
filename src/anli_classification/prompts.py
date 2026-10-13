from __future__ import annotations

from anli_classification.schemas import Example


class NliPrompt:
    """Zero-shot NLI prompt. Bump VERSION when the wording changes so runs never mix."""

    VERSION = "zero-shot-v1"

    SYSTEM = "You are an expert at natural language inference."

    USER = (
        "Premise: {premise}\n"
        "Hypothesis: {hypothesis}\n\n"
        "Given the premise, is the hypothesis definitely true (entailment), "
        "definitely false (contradiction), or neither (neutral)?\n"
        "Answer with exactly one word: entailment, neutral, or contradiction."
    )

    def build(self, example: Example) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.SYSTEM},
            {
                "role": "user",
                "content": self.USER.format(
                    premise=example.premise, hypothesis=example.hypothesis
                ),
            },
        ]
