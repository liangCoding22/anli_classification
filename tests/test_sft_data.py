from anli_classification.sft_data import skip_reason, to_messages
from anli_classification.sft_train import checkpoint_epochs, steps_per_epoch


def trace(gold, pred, reasoning="think"):
    return {
        "key": "train_r3/x", "gold": gold, "label": ["entailment", "neutral", "contradiction"][gold],
        "premise": "p", "hypothesis": "h", "reason": "", "category": "mixed",
        "trace": {"sample": 0, "pred": pred, "raw": pred, "reasoning": reasoning},
    }


def test_assistant_label_is_the_gold_label_and_reasoning_is_stripped():
    msgs = to_messages(trace(2, "contradiction", reasoning="\nthinking\n"))
    assert msgs[-1] == {"role": "assistant", "content": "contradiction", "reasoning_content": "thinking"}


def test_empty_reasoning_and_wrong_answers_are_skipped_with_a_reason():
    assert skip_reason(trace(0, "entailment", reasoning="  \n")) == "empty_reasoning"
    assert skip_reason(trace(0, "neutral")) == "pred_not_gold"
    assert skip_reason(trace(0, "entailment")) is None


def test_steps_and_checkpoint_epochs():
    assert steps_per_epoch(9007, 64) == 141
    assert checkpoint_epochs(141, 2) == {"step-141": 1, "step-282": 2}
