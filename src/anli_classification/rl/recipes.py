"""Named sets of flags, so a known-good run is one flag instead of fifteen.

`--recipe NAME` sets these as the defaults; any flag given explicitly still wins. `legacy` changes nothing (the defaults in
`defaults.py`), so commands without `--recipe` behave as they always did.

`v2` is the last run (`anli-rl-logs/n50000-v2`, 12 steps, dev 0.656 -> 0.694): 16 completions x 48 prompts at 4e-6, the LLM
consistency penalty with the std floor, class-drop and filter-ratio guards, and checkpoint plus eval every 2 steps. It pins
`reward_version` to v2 (the reward that run used). `tests/test_recipes.py` checks this recipe against the run's recorded
`run_config.json`, so it cannot drift from what was actually run. Pass `--pool-path` yourself.
"""

from __future__ import annotations

from typing import Any

RECIPES: dict[str, dict[str, Any]] = {
    "legacy": {},
    "v2": {
        "completions_per_prompt": 16,
        "prompt_groups_per_step": 48,
        "learning_rate": 4e-6,
        "max_steps": 12,
        "checkpoint_interval": 2,
        "eval_interval": 2,
        "dev_rows": 1000,
        "stop_filter_ratio": 0.6,
        "stop_class_drop": 0.10,
        "class_drop_alert": 0.05,
        "consistency_penalty": 0.25,
        "advantage_std_floor": 0.2,
        "reward_version": "v2",
        "weight_sync_timeout": 1800,
        "stall_timeout": 600,
        "skip_cleanup": True,
    },
}
