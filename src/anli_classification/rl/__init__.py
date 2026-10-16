"""GRPO post-training on the training pool, split by concern.

- `defaults`: default values and file names.
- `reward`: the reward of a training rollout (label reward, LLM consistency judge, shaped reward, advantages). No cookbook imports.
- `dev_metrics`, `data`, `logs`: dev accuracy and per-class metrics, pool rows, rollout and format logs.
- `rollout`: the single-turn rollout the cookbook loop calls.
- `guards`, `stop`: run guards and how a run is ended from inside the loop.
- `cookbook_patches`: runtime patches to the cookbook's training library, kept in one place.
- `cleanup`: deleting what a run created.
- `launch`, `cli`: arguments to the cookbook config, and the command line.

`anli_classification.rl_train` re-exports everything, so existing imports and the `anli-rl-train` entry point keep working.
"""
