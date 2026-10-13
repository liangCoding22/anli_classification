from __future__ import annotations

from anli_classification.schemas import Example


class NliPrompt:
    """Zero-shot NLI prompt. Bump VERSION when the wording changes so runs never mix."""

    VERSION = "zero-shot-v2"

    SYSTEM = "You are an expert evaluator analyzing textual relationships. Read the premise and the hypothesis below, then determine the logical relationship between them."

    USER = (
        "Premise: {premise}\n"
        "Hypothesis: {hypothesis}\n\n"
        "Based on the premise, choose exactly one of the following labels for the hypothesis:\n"
        "- entailment (The hypothesis is definitely true given the premise)\n"
        "- contradiction (The hypothesis is definitely false given the premise)\n"
        "- neutral (The hypothesis might be true or false; there is not enough information to decide)\n"
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


class NliPromptV3(NliPrompt):
    """Label definitions in terms of entailment, contradiction, and neutral as logical relations."""

    VERSION = "zero-shot-v3"

    SYSTEM = "You are an expert evaluator analyzing textual relationships. Read the premise and the hypothesis below, then determine their semantic relationship."

    USER = (
        "Premise: {premise}\n"
        "Hypothesis: {hypothesis}\n\n"
        "Based on the premise, choose exactly one of the following labels for the hypothesis:\n\n"
        "- entailment: The hypothesis follows from the premise, including through paraphrasing, implied meaning, or ordinary common sense.\n"
        "- contradiction: The hypothesis conflicts with the premise, including through incompatible facts or meanings.\n"
        "- neutral: The premise neither supports nor contradicts the hypothesis.\n\n"
        "Consider the meaning and context rather than requiring explicit word matches. Do not choose neutral solely because the relationship is implicit.\n"
        "Answer with exactly one word: entailment, neutral, or contradiction."
    )


# Prompt versions selectable from the CLI. Each one writes to its own run directory.
PROMPTS: dict[str, type[NliPrompt]] = {
    NliPrompt.VERSION: NliPrompt,
    NliPromptV3.VERSION: NliPromptV3,
}
