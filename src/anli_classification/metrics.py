from __future__ import annotations

import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)[
            "accuracy"
        ],
        "macro_f1": f1.compute(
            predictions=predictions, references=labels, average="macro"
        )["f1"],
    }
