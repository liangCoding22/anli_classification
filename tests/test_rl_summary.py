import json

from anli_classification.rl_summary import UNAVAILABLE, summarize, write_summary


def write_run(log_dir, batches, evals, rollouts):
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run_config.json").write_text(json.dumps({
        "args": {"completions_per_prompt": "8", "epochs": "2"},
        "train_keys": ["a", "b", "c"],
    }))
    metrics = [
        {"step": i + 1, "train/step": i + 1, "rollout/raw_samples": raw, "rollout/filtered_samples": kept,
         "rollout/completion_tokens_mean": 100.0, "rollout/truncated_ratio": 0.0}
        for i, (raw, kept) in enumerate(batches)
    ] + [{"step": step, "dev/accuracy": acc, "dev/valid_accuracy": acc} for step, acc in evals]
    (log_dir / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
    (log_dir / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rollouts))


def test_acceptance_rate_counts_groups_not_completions(tmp_path):
    # 2 batches of 4 groups x 8 completions attempted; 1 and 3 groups kept.
    write_run(tmp_path, [(32, 8), (32, 24)], [(0, 0.5), (1, 0.6)], [])
    s = summarize(tmp_path, final_checkpoint="step-2")
    assert s["attempted_groups"] == 8
    assert s["accepted_groups"] == 4
    assert s["acceptance_rate"] == 0.5
    assert s["optimizer_updates"] == 2
    assert s["initial_dev_accuracy"] == {"step": 0, "accuracy": 0.5}
    assert s["best_dev_accuracy"] == {"step": 1, "accuracy": 0.6}
    assert s["final_checkpoint"] == "step-2"
    assert s["rejected_all_correct"] == UNAVAILABLE


def test_format_counts_and_missing_data_marked_unavailable(tmp_path):
    rollouts = [
        {"split": "train", "well_formed": True},
        {"split": "train", "well_formed": False},
        {"split": "train", "well_formed": False},
        {"split": "train", "well_formed": True},
        {"split": "eval", "well_formed": True},
    ]
    write_run(tmp_path, [], [], rollouts)
    s = summarize(tmp_path)
    counts = s["train_rollouts"]
    assert (counts["rollouts"], counts["well_formed"], counts["malformed"], counts["malformed_rate"]) == (4, 2, 2, 0.5)
    assert s["eval_rollouts"]["rollouts"] == 1
    assert s["acceptance_rate"] == UNAVAILABLE and s["final_dev_accuracy"] == UNAVAILABLE
    assert "Acceptance rate: unavailable" in s["text"]


def test_write_summary_creates_file(tmp_path):
    write_run(tmp_path, [(16, 16)], [(0, 0.25)], [])
    write_summary(tmp_path)
    assert (tmp_path / "summary.txt").read_text().startswith("Training Summary")


def test_parse_status_counts_are_reported(tmp_path):
    rollouts = [
        {"split": "train", "well_formed": True, "parse_status": "parsed"},
        {"split": "train", "well_formed": False, "parse_status": "unclosed_think"},
        {"split": "train", "well_formed": False, "parse_status": "unclosed_think"},
    ]
    write_run(tmp_path, [], [], rollouts)
    s = summarize(tmp_path)
    assert s["train_rollouts"]["parse_status"] == {"parsed": 1, "unclosed_think": 2}


def test_outcome_counts_separate_incorrect_from_invalid(tmp_path):
    rollouts = [
        {"split": "train", "outcome": "correct", "well_formed": True, "parse_status": "parsed"},
        {"split": "train", "outcome": "incorrect", "well_formed": True, "parse_status": "parsed"},
        {"split": "train", "outcome": "invalid", "well_formed": False, "parse_status": "unclosed_think"},
    ]
    write_run(tmp_path, [], [], rollouts)
    s = summarize(tmp_path)
    assert s["train_rollouts"]["outcome"] == {"correct": 1, "incorrect": 1, "invalid": 1}
