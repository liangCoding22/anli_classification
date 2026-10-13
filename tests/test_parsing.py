from anli_classification.parsers import LabelParser
from anli_classification.prompts import NliPrompt
from anli_classification.schemas import LABELS, LABEL2ID, Example


def test_label_ids_follow_label_order():
    assert LABELS == ("entailment", "neutral", "contradiction")
    assert LABEL2ID["contradiction"] == 2


def test_parser_handles_exact_and_noisy_replies():
    parser = LabelParser()
    assert parser.parse("entailment") == 0
    assert parser.parse("Neutral.") == 1
    assert parser.parse("  CONTRADICTION\n") == 2
    assert parser.parse("The answer is: contradiction") == 2


def test_parser_returns_none_when_no_label_named():
    parser = LabelParser()
    assert parser.parse("") is None
    assert parser.parse(None) is None
    assert parser.parse("I am not sure.") is None
    assert parser.parse("entailments") is None  # whole-word match only


def test_prompt_embeds_premise_and_hypothesis():
    example = Example("dev", "r1", "u1", "A dog runs.", "An animal moves.", 0)
    messages = NliPrompt().build(example)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Premise: A dog runs." in messages[1]["content"]
    assert "Hypothesis: An animal moves." in messages[1]["content"]
