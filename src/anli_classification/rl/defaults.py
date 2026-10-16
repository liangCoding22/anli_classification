"""Defaults and file names for the GRPO driver."""

from __future__ import annotations


FORMAT_SNAPSHOT_EVERY = 512  # training rollouts per format snapshot: 64 prompt groups x 8 completions


METRICS_FILE = "metrics.jsonl"  # every metric the cookbook logs, one JSON line per call


ROLLOUTS_FILE = "rollouts.jsonl"  # every graded rollout: key, gold, pred, reward, completion


CONFIG_FILE = "run_config.json"  # arguments and example keys the run started with


EPOCHS = 1  # passes over the training rows; rejected rows also use draws, so optimizer steps are fewer


COMPLETIONS_PER_PROMPT = 8  # rollouts per prompt (the GRPO group size)


PROMPT_GROUPS_PER_STEP = 64  # prompts per optimizer step


LEARNING_RATE = 1e-5


TEMPERATURE = 1.0


EVAL_TEMPERATURE = 0.0  # dev evaluation is greedy; the cookbook would reuse the training temperature


MAX_COMPLETION_TOKENS = 4096


DEV_ROWS = 1000  # about 330 per class, so per-class recall has a standard error near 2.6 points


JUDGE_MODEL = "accounts/fireworks/models/gpt-oss-120b"  # serverless; tested on 120 real rollouts, 4 flagged, all genuine


JUDGE_MAX_CHARS = 16000  # reasoning longer than this keeps its first 3000 and last 12000 characters


REWARD_VERSIONS = ("v2", "v3")  # v2: the reward of the v2 run (kept as the default); v3: see training_reward


MALFORMED_MARGIN = 0.25  # v3: a malformed answer scores this far below the worst parseable outcome


NO_REASONING = "no_reasoning"  # verdict recorded for a parseable answer that has no reasoning text (v3)


ADVANTAGE_STD_FLOOR = 0.2  # binary groups with >= 1 success in 16 have std >= 0.25, so only penalty-only groups are affected


CLASS_DROP_ALERT = 0.05  # alert when a class's dev recall falls this far below the run's first evaluation


DCP_SAVE_INTERVAL = 2  # resumable checkpoint every 2 optimizer steps


# Metrics to watch in the logs or W&B (names as the cookbook's async reference documents them;
# check them against the first step's logged keys in case the sync loop names them differently).
MONITOR_METRICS = (
    "rollout/raw_reward",  # mean reward before filtering; should rise
    "rollout/filtered_reward",  # mean reward on groups that survived the filter
    "rollout/filter_ratio",  # share of groups dropped for zero reward variance; should grow
    "rollout/truncated_ratio",  # completions hitting max tokens; should stay near 0
    "rollout/completion_tokens_mean",
    "train/ppo_kl",  # policy drift per step; a sudden jump means the learning rate is too high
    "train/ppo_clip_frac",
    "train/grad_norm",
)


TRAINER_JOB_FILE = "trainer_job.json"  # trainer job id, needed to list and promote checkpoints later


SAMPLE_TIMEOUT_S = 300  # per sampling request; a hung connection should fail in minutes, not 20


STALL_TIMEOUT_S = 1200  # kill the run when neither rollouts.jsonl nor metrics.jsonl changes for this long


WEIGHT_SYNC_TIMEOUT_S = 1800  # how long the cookbook waits for the deployment to report a new snapshot (default 600)
