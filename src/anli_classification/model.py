from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from anli_classification import ID2LABEL, LABEL2ID, NUM_LABELS


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name: str):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
