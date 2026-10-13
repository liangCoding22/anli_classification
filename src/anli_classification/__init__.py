"""ANLI 3-way natural language inference classification."""

LABEL2ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

__all__ = ["ID2LABEL", "LABEL2ID", "NUM_LABELS"]
