import json

from types import SimpleNamespace

from anli_classification.rl_train import dev_accuracy, label_reward, load_pool_rows


def test_label_reward_is_one_only_for_the_gold_label():
    assert label_reward("entailment", 0) == 1.0
    assert label_reward("Neutral", 1) == 1.0
    assert label_reward("Neutral.", 1) == 0.0  # trailing punctuation is not allowed
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


def run(reward):
    return SimpleNamespace(segments=[SimpleNamespace(reward=reward)])


def test_dev_accuracy_counts_failed_rollouts_as_incorrect():
    rows = [{"key": "a"}, {"key": "b"}, {"key": "c"}, {"key": "d"}]
    runs_by_row = [[run(1.0)], [run(0.0)], [None], [run(1.0)]]  # one failed rollout
    metrics = dev_accuracy(rows, runs_by_row)
    assert metrics["dev/accuracy"] == 2 / 4  # failed row is in the denominator
    assert metrics["dev/valid_accuracy"] == 2 / 3  # over the three scored rollouts
    assert metrics["dev/dropped_ratio"] == 1 / 4


def test_dev_accuracy_with_no_failures_matches_both_metrics():
    rows = [{"key": "a"}, {"key": "b"}]
    metrics = dev_accuracy(rows, [[run(1.0)], [run(0.0)]])
    assert metrics["dev/accuracy"] == metrics["dev/valid_accuracy"] == 0.5
    assert metrics["dev/dropped_ratio"] == 0.0


def test_label_reward_missing_answer_is_zero():
    assert label_reward("", 0) == 0.0
    assert label_reward("The premise says nothing useful", 1) == 0.0


def test_label_reward_truncated_reasoning_without_label_is_zero():
    truncated = "Let me think. The hypothesis could follow from the premise if we assume"
    assert label_reward(truncated, 0) == 0.0


def test_label_reward_requires_exactly_one_label_in_the_answer():
    assert label_reward("neutral", 1) == 1.0
    assert label_reward("Entailment. Not neutral.", 0) == 0.0  # not exactly one label word
    assert label_reward("neutral\n\nexplanation after the label", 1) == 0.0
    assert label_reward("<think>entailment</think>neutral", 1) == 1.0
    assert label_reward("<think>neutral, but unclosed", 1) == 0.0


def test_thinking_text_extracts_only_reasoning_parts():
    from anli_classification.rl_train import thinking_text

    message = {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "maybe neutral"},
        {"type": "text", "text": "entailment"},
    ]}
    assert thinking_text(message) == "maybe neutral"
    assert thinking_text({"role": "assistant", "content": "entailment"}) == ""


def test_is_well_formed_accepts_only_one_label_word():
    from anli_classification.parsers import is_well_formed

    assert is_well_formed("entailment")
    assert is_well_formed("  Neutral\n")  # case and surrounding whitespace are ignored
    assert is_well_formed("CONTRADICTION")
    assert not is_well_formed("neutral\n\nThe premise says so.")  # label then explanation
    assert not is_well_formed("neutral.")  # trailing punctuation is not one label word
    assert not is_well_formed("entailed, neutral")
    assert not is_well_formed("neutral\n\nWait, the earlier analysis said entailment.")
    assert not is_well_formed("")


def test_format_tracker_writes_snapshots_and_alerts(tmp_path, capsys):
    from anli_classification.rl_train import FormatTracker

    path = tmp_path / "metrics.jsonl"
    tracker = FormatTracker(path, window=4, alert_rate=0.25)
    for well in [True, True, False, True]:  # 1 of 4 malformed: 25%, not above the threshold
        tracker.record(well)
    assert "ALERT" not in capsys.readouterr().out
    for well in [True, False, True]:  # 1 of 3 malformed in the partial window: above the threshold
        tracker.record(well)
    tracker.flush()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["type"] == "format" and "step" not in records[0]
    assert records[0]["format/malformed_rate"] == 0.25
    assert records[1]["window_rollouts"] == 3
    assert "ALERT" in capsys.readouterr().out


def test_dev_strict_accuracy_counts_malformed_answers_as_wrong():
    rows = [{"key": "a"}, {"key": "b"}, {"key": "c"}]
    runs_by_row = [[run(1.0)], [run(1.0)], [run(0.0)]]
    formats = {"a": "parsed", "b": "not_one_label", "c": "unclosed_think"}
    metrics = dev_accuracy(rows, runs_by_row, formats=formats)
    assert metrics["dev/accuracy"] == 2 / 3
    assert metrics["dev/strict_accuracy"] == 1 / 3
    assert metrics["dev/invalid_ratio"] == 2 / 3
    assert formats == {}


def test_max_steps_must_match_checkpoint_and_eval_intervals(capsys):
    import pytest as _pytest

    from anli_classification.rl_train import parse_args

    with _pytest.raises(SystemExit):
        parse_args(["--pool-path", "x", "--max-steps", "5", "--checkpoint-interval", "2", "--eval-interval", "2"])
    args = parse_args(["--pool-path", "x", "--max-steps", "6", "--checkpoint-interval", "2", "--eval-interval", "2"])
    assert args.max_steps == 6


def test_intermediate_checkpoint_names():
    from anli_classification.rl_train import is_intermediate_checkpoint

    assert is_intermediate_checkpoint("step-2")
    assert is_intermediate_checkpoint("step-12")
    assert not is_intermediate_checkpoint("step-2-8377bd45")  # server names, not caller names
    assert not is_intermediate_checkpoint("final-merged")


def test_unparsed_reasons_are_distinct():
    from anli_classification.parsers import unparsed_reason

    assert unparsed_reason("neutral") is None
    assert unparsed_reason("") == "empty"
    assert unparsed_reason("neutral because the premise is silent") == "not_one_label"  # fullmatch, not search
    assert unparsed_reason("<think>x</think>") == "empty_answer"
    assert unparsed_reason("<think>x") == "unclosed_think"
    assert unparsed_reason("<think>a <think>b</think>neutral") == "unbalanced_think"
    assert unparsed_reason("<think>a</think>neutral<think>partial") == "unclosed_think"
    assert unparsed_reason("<think>a</think>neutral") is None
    assert unparsed_reason("<think>a</think>\n<think>b</think>\nneutral") is None  # several closed blocks


def test_unparsed_answers_count_as_incorrect_not_dropped():
    rows = [{"key": "a"}, {"key": "b"}]
    runs_by_row = [[run(0.0)], [run(1.0)]]  # "a" was unparsed: reward 0
    metrics = dev_accuracy(rows, runs_by_row, formats={"a": "not_one_label", "b": "parsed"})
    assert metrics["dev/accuracy"] == 1 / 2
    assert metrics["dev/dropped_ratio"] == 0.0
    assert metrics["dev/strict_accuracy"] == 1 / 2
    assert metrics["dev/invalid_ratio"] == 1 / 2


def _eval_hook(tmp_path, ratios, threshold, monkeypatch):
    import asyncio
    import sys
    from types import ModuleType

    from anli_classification.rl_train import StopAtStep, print_eval

    logged = []
    fake = ModuleType("training.utils.logging")
    fake.log_metrics = lambda metrics, step: logged.append(step)
    monkeypatch.setitem(sys.modules, "training.utils.logging", fake)
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text("".join(json.dumps({"step": i, "rollout/filter_ratio": r}) + "\n"
                                    for i, r in enumerate(ratios, 1)))

    async def evaluation(step, rollout_fn):
        return {"dev/accuracy": 0.5}

    hook = print_eval(evaluation, stop_filter_ratio=threshold, metrics_path=metrics_path)
    try:
        asyncio.run(hook(len(ratios), None))
    except StopAtStep as stop:
        return stop.step, logged
    return None, logged


def test_stop_filter_ratio_stops_when_recent_steps_are_mostly_filtered(tmp_path, monkeypatch):
    stopped, logged = _eval_hook(tmp_path, [0.1, 0.6, 0.7, 0.55], 0.5, monkeypatch)
    assert stopped == 4 and logged == [4]


def test_stop_filter_ratio_ignores_early_spike_and_short_history(tmp_path, monkeypatch):
    assert _eval_hook(tmp_path, [0.9, 0.2, 0.3, 0.3], 0.5, monkeypatch)[0] is None  # mean of last 3 is 0.27
    assert _eval_hook(tmp_path, [0.9, 0.9], 0.5, monkeypatch)[0] is None  # fewer than 3 steps
    assert _eval_hook(tmp_path, [0.9, 0.9, 0.9], None, monkeypatch)[0] is None  # guard off


def test_delete_run_resources_deletes_trainer_and_only_new_deployments_of_the_base_model(tmp_path, monkeypatch):
    import time

    from anli_classification.rl_train import delete_run_resources

    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    (tmp_path / "trainer_job.json").write_text(json.dumps({"trainer_job_id": "job1"}))
    started = time.time()
    fresh = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started + 5))
    stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started - 3600))
    deployments = [
        {"name": "accounts/a/deployments/new", "baseModel": "m", "createTime": fresh},
        {"name": "accounts/a/deployments/old", "baseModel": "m", "createTime": stale},
        {"name": "accounts/a/deployments/other", "baseModel": "x", "createTime": fresh},
    ]
    calls = []

    def api(method, path, api_key, base_url):
        calls.append((method, path))
        if path == "accounts":
            return {"accounts": [{"name": "accounts/a"}]}
        if method == "GET":
            return {"deployments": deployments}
        return {}

    done = delete_run_resources(tmp_path, started, "m", api=api)
    deletes = [path for method, path in calls if method == "DELETE"]
    assert deletes == ["accounts/a/rlorTrainerJobs/job1", "accounts/a/deployments/new?ignoreChecks=true"]
    assert len(done) == 2


def test_stall_watchdog_fires_only_after_the_timeout_without_file_activity(tmp_path):
    from anli_classification.rl_train import StallWatchdog

    now = [1000.0]
    log = tmp_path / "rollouts.jsonl"
    dog = StallWatchdog([log], timeout=600, clock=lambda: now[0])
    now[0] = 1500.0
    assert not dog.stalled()  # 500 s since start, under the timeout
    now[0] = 1700.0
    assert dog.stalled()  # 700 s with no file at all
    log.write_text("x")
    import os

    os.utime(log, (1650.0, 1650.0))
    assert not dog.stalled()  # activity at 1650 s resets the clock
    now[0] = 2300.0
    assert dog.stalled()


def test_delete_run_resources_also_deletes_a_reattached_older_deployment(tmp_path, monkeypatch):
    import time

    from anli_classification.rl_train import delete_run_resources

    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    started = time.time()
    stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started - 3600))
    calls = []

    def api(method, path, api_key, base_url):
        calls.append((method, path))
        if path == "accounts":
            return {"accounts": [{"name": "accounts/a"}]}
        if method == "GET":
            return {"deployments": [{"name": "accounts/a/deployments/mine", "baseModel": "m", "createTime": stale}]}
        return {}

    delete_run_resources(tmp_path, started, "m", api=api, deployment_id="mine")
    assert [path for method, path in calls if method == "DELETE"] == ["accounts/a/deployments/mine?ignoreChecks=true"]


def test_dev_accuracy_reports_per_class_recall_and_hides_nothing_when_accuracy_is_flat():
    rows = [{"key": f"r{i}", "gold": g} for i, g in enumerate([0, 0, 1, 1, 2, 2])]
    runs = [[run(1.0)], [run(0.0)], [run(1.0)], [run(1.0)], [run(0.0)], [run(0.0)]]
    preds = {"r0": 0, "r1": 1, "r2": 1, "r3": 1, "r4": 1, "r5": None}
    metrics = dev_accuracy(rows, runs, preds=preds)
    assert metrics["dev/recall_entailment"] == 0.5
    assert metrics["dev/recall_neutral"] == 1.0
    assert metrics["dev/recall_contradiction"] == 0.0
    assert metrics["dev/predicted_share_neutral"] == 4 / 6
    # entailment: precision 1, recall 0.5 -> F1 2/3; neutral: precision 2/4, recall 1 -> F1 2/3; contradiction: 0
    assert abs(metrics["dev/macro_f1"] - (2 / 3 + 2 / 3 + 0.0) / 3) < 1e-9
    assert preds == {}


def test_failed_rollouts_count_as_no_prediction():
    rows = [{"key": "a", "gold": 0}, {"key": "b", "gold": 0}]
    metrics = dev_accuracy(rows, [[run(1.0)], []], preds={"a": 0, "b": 0})
    assert metrics["dev/recall_entailment"] == 0.5


def test_class_drop_alert_fires_only_when_a_class_falls_below_the_first_evaluation(capsys):
    import asyncio

    from anli_classification.rl_train import print_eval

    recalls = iter([0.74, 0.70, 0.65])

    async def evaluation(step, rollout_fn):
        return {"dev/recall_entailment": next(recalls)}

    hook = print_eval(evaluation, class_drop_alert=0.05)
    for step in (0, 5, 10):
        asyncio.run(hook(step, None))
    alerts = [line for line in capsys.readouterr().out.splitlines() if line.startswith("ALERT")]
    assert len(alerts) == 1 and "entailment" in alerts[0] and "step 10" in alerts[0]


def test_delete_run_resources_finds_a_pending_trainer_when_no_job_file_was_written(tmp_path, monkeypatch):
    import time

    from anli_classification.rl_train import delete_run_resources

    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    started = time.time()
    fresh = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started + 5))
    older = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started - 1100))  # someone else's job, 18 minutes earlier
    jobs = [
        {"name": "accounts/a/rlorTrainerJobs/mine", "state": "JOB_STATE_PENDING", "createTime": fresh,
         "trainingConfig": {"baseModel": "m"}},
        {"name": "accounts/a/rlorTrainerJobs/other", "state": "JOB_STATE_RUNNING", "createTime": older,
         "trainingConfig": {"baseModel": "m"}},
        {"name": "accounts/a/rlorTrainerJobs/gone", "state": "JOB_STATE_DELETED", "createTime": fresh,
         "trainingConfig": {"baseModel": "m"}},
    ]
    calls = []

    def api(method, path, api_key, base_url):
        calls.append((method, path))
        if path == "accounts":
            return {"accounts": [{"name": "accounts/a"}]}
        if method == "GET" and "rlorTrainerJobs" in path:
            return {"rlorTrainerJobs": jobs}
        if method == "GET":
            return {"deployments": []}
        return {}

    delete_run_resources(tmp_path, started, "m", api=api)
    assert [path for method, path in calls if method == "DELETE"] == ["accounts/a/rlorTrainerJobs/mine"]


def test_parse_judge_reply_reads_only_the_verdict_and_does_not_confuse_inconsistent_with_consistent():
    from anli_classification.rl_train import parse_judge_reply

    assert parse_judge_reply("inconsistent neutral") == "inconsistent"
    assert parse_judge_reply("Consistent entailment") == "consistent"
    assert parse_judge_reply("  unclear none") == "unclear"
    assert parse_judge_reply("The reasoning is consistent") is None
    assert parse_judge_reply(None) is None and parse_judge_reply("") is None


def test_shaped_reward_penalizes_only_an_inconsistent_verdict():
    from anli_classification.rl_train import shaped_reward

    assert shaped_reward(1.0, "inconsistent", 0.25) == 0.75
    assert shaped_reward(0.0, "inconsistent", 0.25) == -0.25
    assert shaped_reward(1.0, "consistent", 0.25) == 1.0
    assert shaped_reward(1.0, "unclear", 0.25) == 1.0
    assert shaped_reward(1.0, None, 0.25) == 1.0


def test_judge_sees_premise_hypothesis_reasoning_and_label_but_not_gold():
    from anli_classification.rl_train import ConsistencyJudge

    seen = []

    def post(body):
        seen.append(body)
        return {"choices": [{"message": {"content": "inconsistent neutral"}}]}

    judge = ConsistencyJudge(post=post)
    verdict = judge.verdict("A dog runs.", "An animal moves.", "reasoning ... so the answer is neutral", "entailment")
    assert verdict == "inconsistent"
    user = seen[0]["messages"][1]["content"]
    assert "A dog runs." in user and "An animal moves." in user and "so the answer is neutral" in user
    assert "Label the model output: entailment" in user
    assert "gold" not in user.lower()


def test_judge_fails_open_and_caches():
    from anli_classification.rl_train import ConsistencyJudge

    calls = []

    def broken(body):
        calls.append(1)
        raise OSError("network down")

    judge = ConsistencyJudge(post=broken, report_every=10 ** 9)
    import anli_classification.rl_train as module

    original_sleep = module.time.sleep
    module.time.sleep = lambda _: None
    try:
        assert judge.verdict("p", "h", "t", "neutral") is None  # an outage gives no verdict, hence no penalty
    finally:
        module.time.sleep = original_sleep
    assert len(calls) == 3 and judge.stats["errors"] == 1

    ok = ConsistencyJudge(post=lambda body: {"choices": [{"message": {"content": "consistent entailment"}}]}, report_every=10 ** 9)
    assert ok.verdict("p", "h", "t", "entailment") == "consistent"
    assert ok.verdict("p", "h", "t", "entailment") == "consistent"
    assert ok.stats["cache_hits"] == 1


def test_load_pool_rows_carries_premise_and_hypothesis_for_the_judge(tmp_path):
    from anli_classification.rl_train import load_pool_rows

    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({"key": "train_r3/abc", "gold": 1, "premise": "P text", "hypothesis": "H text"}) + "\n")
    row = load_pool_rows(pool)[0]
    assert row["premise"] == "P text" and row["hypothesis"] == "H text"


def test_stop_class_drop_stops_the_run_at_the_eval_where_a_class_collapses(capsys, monkeypatch):
    import asyncio
    import sys
    from types import ModuleType

    from anli_classification.rl_train import StopAtStep, print_eval

    fake = ModuleType("training.utils.logging")
    fake.log_metrics = lambda metrics, step: None
    monkeypatch.setitem(sys.modules, "training.utils.logging", fake)
    recalls = iter([0.70, 0.62, 0.55])

    async def evaluation(step, rollout_fn):
        return {"dev/recall_neutral": next(recalls)}

    hook = print_eval(evaluation, stop_class_drop=0.10)
    asyncio.run(hook(0, None))
    asyncio.run(hook(2, None))  # 0.08 below the baseline: under the limit
    try:
        asyncio.run(hook(4, None))
    except StopAtStep as stop:
        assert stop.step == 4
    else:
        raise AssertionError("expected a stop")
    assert "neutral dev recall fell" in capsys.readouterr().out


def test_floored_advantages_keep_a_lone_penalty_in_proportion_and_leave_binary_groups_alone():
    from anli_classification.rl_train import floored_advantages

    # 15 correct and consistent, 1 correct but penalised 0.25: plain z-scoring would give about -3.75
    group = [1.0] * 15 + [0.75]
    plain = floored_advantages(group, std_floor=0.0)
    floored = floored_advantages(group, std_floor=0.2)
    assert abs(plain[-1] - (-3.75)) < 0.01 and abs(plain[0] - 0.25) < 0.01
    assert abs(floored[-1] - (-0.234375 / 0.2)) < 0.01  # about -1.17
    # a smaller penalty now gives a smaller push; without the floor it would not
    assert abs(floored_advantages([1.0] * 15 + [0.9], std_floor=0.2)[-1]) < abs(floored[-1])
    assert abs(floored_advantages([1.0] * 15 + [0.9], std_floor=0.0)[-1] - plain[-1]) < 0.01
    # ordinary binary groups have std >= 0.25, so the floor changes nothing
    binary = [1.0] * 8 + [0.0] * 8
    assert floored_advantages(binary, 0.2) == floored_advantages(binary, 0.0)
    one_success = [1.0] + [0.0] * 15
    assert floored_advantages(one_success, 0.2) == floored_advantages(one_success, 0.0)
    # equal rewards stay at zero
    assert floored_advantages([1.0] * 16, 0.2) == [0.0] * 16
