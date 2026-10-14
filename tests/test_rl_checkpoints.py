import pytest

from anli_classification.rl_checkpoints import find_row, format_rows, promote, short_name, validate_output_model_id


ROWS = [
    {"name": "accounts/a/rlorTrainerJobs/job1/checkpoints/step-0-abc", "promotable": True,
     "checkpointType": "CHECKPOINT_TYPE_INFERENCE_BASE", "createTime": "t0"},
    {"name": "accounts/a/rlorTrainerJobs/job1/checkpoints/step-2-abc", "promotable": True,
     "checkpointType": "CHECKPOINT_TYPE_INFERENCE_BASE", "createTime": "t2"},
]


class FakeManager:
    def __init__(self, rows):
        self.rows = rows
        self.promoted = []

    def list_checkpoints(self, job_id):
        return self.rows

    def promote_checkpoint(self, **kwargs):
        self.promoted.append(kwargs)
        return {"state": "READY", "name": kwargs["output_model_id"]}


def test_short_name_and_find_row():
    assert short_name(ROWS[1]) == "step-2-abc"
    assert find_row(ROWS, "step-2-abc") is ROWS[1]
    with pytest.raises(SystemExit):
        find_row(ROWS, "step-9-abc")


def test_promote_uses_full_name_of_promotable_row():
    manager = FakeManager(ROWS)
    result = promote("job1", "step-2-abc", "anli-step2", "accounts/fireworks/models/qwen3-4b", manager=manager)
    assert manager.promoted == [{"name": ROWS[1]["name"], "output_model_id": "anli-step2",
                                 "base_model": "accounts/fireworks/models/qwen3-4b"}]
    assert result["state"] == "READY"


def test_promote_refuses_non_promotable_rows():
    rows = [dict(ROWS[1], promotable=False, checkpointType="CHECKPOINT_TYPE_INFERENCE_ARC_V2")]
    with pytest.raises(SystemExit):
        promote("job1", "step-2-abc", "anli-step2", "m", manager=FakeManager(rows))


def test_output_model_id_rules():
    validate_output_model_id("anli-qwen3-4b-step-2")
    for bad in ["Has-Caps", "under_score", "x" * 64]:
        with pytest.raises(SystemExit):
            validate_output_model_id(bad)


def test_format_rows_lists_promotable_flag():
    text = format_rows(ROWS)
    assert "step-0-abc" in text and "True" in text
