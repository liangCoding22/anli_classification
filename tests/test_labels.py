from anli_classification import ID2LABEL, LABEL2ID, NUM_LABELS


def test_label_maps_are_inverses():
    assert NUM_LABELS == 3
    assert LABEL2ID["entailment"] == 0
    assert ID2LABEL[LABEL2ID["neutral"]] == "neutral"
    assert set(LABEL2ID) == {"entailment", "neutral", "contradiction"}
