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


class NliPromptV4A(NliPromptV3):
    """v3 with a wider contradiction definition and a stricter neutral: 'not mentioned' is not enough for neutral."""

    VERSION = "zero-shot-v4a"

    USER = (
        "Premise: {premise}\n"
        "Hypothesis: {hypothesis}\n\n"
        "Based on the premise, choose exactly one of the following labels for the hypothesis:\n\n"
        "- entailment: The hypothesis follows from the premise, including through paraphrasing, implied meaning, or ordinary common sense.\n"
        "- contradiction: The hypothesis conflicts with the premise: it asserts something incompatible with what the premise states or clearly implies, "
        "such as a different fact, number, time or state, or something that ordinary common sense rules out given the premise.\n"
        "- neutral: The hypothesis is compatible with the premise but not established by it: for all the premise says, it could easily be true or false.\n\n"
        "Consider the meaning and context rather than requiring explicit word matches. Do not choose neutral solely because the relationship is implicit, "
        "or because the premise does not mention the claim: if the premise makes the claim false or implausible, choose contradiction.\n"
        "Answer with exactly one word: entailment, neutral, or contradiction."
    )


class NliPromptV4B(NliPromptV3):
    """v3 with an explicit order of questions: entailment, then contradiction, then neutral as the remainder."""

    VERSION = "zero-shot-v4b"

    USER = (
        "Premise: {premise}\n"
        "Hypothesis: {hypothesis}\n\n"
        "Based on the premise, choose exactly one label for the hypothesis. Work through these questions in order:\n"
        "1. Does the hypothesis follow from the premise, directly, through paraphrase, implied meaning, or ordinary common sense? If yes: entailment.\n"
        "2. Could the hypothesis be true at the same time as the premise? If the premise makes it false or very unlikely "
        "(a different fact, the opposite claim, or something ordinary common sense rules out): contradiction.\n"
        "3. Otherwise the hypothesis is compatible with the premise but not established by it: neutral.\n\n"
        "Consider the meaning and context rather than requiring explicit word matches.\n"
        "Answer with exactly one word: entailment, neutral, or contradiction."
    )


# Prompt versions selectable from the CLI. Each one writes to its own run directory.
PROMPTS: dict[str, type[NliPrompt]] = {
    NliPrompt.VERSION: NliPrompt,
    NliPromptV3.VERSION: NliPromptV3,
    NliPromptV4A.VERSION: NliPromptV4A,
    NliPromptV4B.VERSION: NliPromptV4B,
}
