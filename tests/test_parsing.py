from anli_classification.parsers import LabelParser
from anli_classification.prompts import NliPrompt
from anli_classification.schemas import LABELS, LABEL2ID, Example


def test_label_ids_follow_label_order():
    assert LABELS == ("entailment", "neutral", "contradiction")
    assert LABEL2ID["contradiction"] == 2


def test_parser_accepts_only_one_label_word_case_insensitively():
    parser = LabelParser()
    assert parser.parse("entailment") == 0
    assert parser.parse("  CONTRADICTION\n") == 2
    assert parser.parse("Neutral") == 1
    assert parser.parse("Neutral.") is None  # trailing punctuation is not allowed
    assert parser.parse("The answer is: contradiction") is None
    assert parser.parse("neutral\n\nThe premise says so.") is None  # label then explanation


def test_parser_reads_only_the_answer_after_a_closed_think_block():
    parser = LabelParser()
    assert parser.parse("<think>entailment? no, neutral</think>\n\nneutral") == 1
    assert parser.parse("<think>a</think>neutral<think>partial entailment") is None
    assert parser.parse("<think>a</think>") is None  # no answer after the thinking


def test_parser_accepts_a_lone_closing_think_tag():
    # Thinking-only models (Qwen3 *-Thinking-2507): the template opens the block, so only </think> appears.
    parser = LabelParser()
    assert parser.parse("Let me think. It is neutral.\n</think>\n\nneutral") == 1
    assert parser.parse("reasoning</think>") is None  # no answer after the thinking
    assert parser.parse("a</think>neutral</think>neutral") is None  # two closing tags
    assert parser.parse("a</think>neutral<think>b") is None
    from anli_classification.parsers import split_reasoning
    assert split_reasoning("why\n</think>\n\nentailment") == ("why\n", "\n\nentailment")
    assert split_reasoning("<think>why</think>entailment") == ("why", "entailment")
    assert split_reasoning("entailment") == ("", "entailment")


def test_parser_returns_none_without_a_valid_answer():
    parser = LabelParser()
    assert parser.parse("") is None
    assert parser.parse(None) is None
    assert parser.parse("I am not sure.") is None
    assert parser.parse("entailments") is None
    # an unclosed thinking block must not yield a label from its reasoning
    assert parser.parse("<think>The answer is entailment, I think") is None


def test_prompt_embeds_premise_and_hypothesis():
    example = Example("dev", "r1", "u1", "A dog runs.", "An animal moves.", 0)
    messages = NliPrompt().build(example)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Premise: A dog runs." in messages[1]["content"]
    assert "Hypothesis: An animal moves." in messages[1]["content"]
