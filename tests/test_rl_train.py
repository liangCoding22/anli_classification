import json

from anli_classification.rl_train import label_reward, load_pool_rows


def test_label_reward_is_one_only_for_the_gold_label():
    assert label_reward("entailment", 0) == 1.0
    assert label_reward("Neutral.", 1) == 1.0
    assert label_reward("contradiction", 0) == 0.0
    assert label_reward("I am not sure", 0) == 0.0


def test_load_pool_rows_builds_messages_and_gold(tmp_path):
    record = {
        "key": "train_r3/u1",
        "group": "1-3/4",
        "gold": 2,
        "premise": "A man sleeps.",
        "hypothesis": "A man is awake.",
        "reason": "",
        "n_correct": 3,
        "samples": [],
    }
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps(record) + "\n")
    rows = load_pool_rows(pool)
    assert len(rows) == 1
    assert rows[0]["gold"] == 2 and rows[0]["key"] == "train_r3/u1"
    assert rows[0]["messages"][0]["role"] == "system"
    assert "A man sleeps." in rows[0]["messages"][1]["content"]
