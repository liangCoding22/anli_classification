"""GRPO post-training on the training pool, with the Fireworks cookbook's strict on-policy loop.

The code now lives in the `anli_classification.rl` package, split by concern (see its docstring). This module
re-exports every name it used to define, so `from anli_classification.rl_train import ...` and the `anli-rl-train`
entry point keep working. New code should import from `anli_classification.rl.<module>`.
"""

from __future__ import annotations

from anli_classification.rl.cleanup import (  # noqa: F401
    _fireworks_api,
    delete_run_resources,
    start_stall_watchdog,
)
from anli_classification.rl.cli import (  # noqa: F401
    main,
    parse_args,
    run_rl,
)
from anli_classification.rl.cookbook_patches import (  # noqa: F401
    _INTERMEDIATE_NAME,
    HOTLOAD_REQUEST_TIMEOUT_S,
    install_checkpoint_patches,
    install_hotload_timeout_patch,
    is_intermediate_checkpoint,
)
from anli_classification.rl.data import (  # noqa: F401
    load_dev_rows,
    load_pool_rows,
)
from anli_classification.rl.defaults import (  # noqa: F401
    ADVANTAGE_STD_FLOOR,
    CLASS_DROP_ALERT,
    COMPLETIONS_PER_PROMPT,
    CONFIG_FILE,
    DCP_SAVE_INTERVAL,
    DEV_ROWS,
    EPOCHS,
    EVAL_TEMPERATURE,
    FORMAT_SNAPSHOT_EVERY,
    JUDGE_MAX_CHARS,
    JUDGE_MODEL,
    LEARNING_RATE,
    MALFORMED_MARGIN,
    MAX_COMPLETION_TOKENS,
    METRICS_FILE,
    MONITOR_METRICS,
    NO_REASONING,
    PROMPT_GROUPS_PER_STEP,
    REWARD_VERSIONS,
    ROLLOUTS_FILE,
    SAMPLE_TIMEOUT_S,
    STALL_TIMEOUT_S,
    TEMPERATURE,
    TRAINER_JOB_FILE,
    WEIGHT_SYNC_TIMEOUT_S,
)
from anli_classification.rl.dev_metrics import (  # noqa: F401
    class_metrics,
    dev_accuracy,
)
from anli_classification.rl.guards import (  # noqa: F401
    print_eval,
    recent_filter_ratio,
    StallWatchdog,
)
from anli_classification.rl.launch import (  # noqa: F401
    build_config,
    resolve_training_shape,
    write_run_config,
)
from anli_classification.rl.logs import (  # noqa: F401
    FormatTracker,
    RolloutLog,
)
from anli_classification.rl.reward import (  # noqa: F401
    _JUDGE_VERDICT,
    ConsistencyJudge,
    floored_advantages,
    judge_prompt,
    JUDGE_SYSTEM,
    label_reward,
    parse_judge_reply,
    shaped_reward,
    thinking_text,
    training_reward,
)
from anli_classification.rl.rollout import (  # noqa: F401
    make_rollout_fn,
)
from anli_classification.rl.stop import (  # noqa: F401
    StopAtStep,
)

if __name__ == "__main__":
    main()
