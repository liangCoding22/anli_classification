# Training (GRPO)

The pipeline that builds the pools, the RL loop and its guards, and the implementation of the last run. Results and what was learned are in [experiments.md](experiments.md); failure modes and cleanup are in [operations.md](operations.md).

## Pipeline components

| Piece | What it does |
|---|---|
| `anli-training-pool` | Samples the 4B on `train_r3` and writes `pool.jsonl` (one line per prompt, with sampled answers). |
| `scripts/select_teacher_prompts.py`, `scripts/generate_teacher_traces.py` | Pick prompts by how hard the 4B found them and collect traces from the teacher (`qwen3p5-397b-a17b`, 2 votes per prompt). |
| `scripts/build_clean_pool.py` | Filters the pool by teacher agreement, balances classes, and builds `pool_v2.jsonl` (see below). |
| `anli-rl-train` (`rl_train.py`) | GRPO on a pool: reward, guards, per-class dev metrics, LLM consistency judge. |
| `anli-rl-summary`, `anli-rl-checkpoints` | Summarize a run; list a trainer's checkpoints and which are promotable. |
| `scripts/promote_checkpoint.py` | Turn one promotable checkpoint into a model. |
| `scripts/eval_promoted.py` | Deploy a promoted model on demand, run `anli-eval --template client` on it, delete the deployment. |

Setup for training (Python 3.12 venv, the Training API is private preview):

```bash
git clone https://github.com/fw-ai/cookbook.git      # used: commit 4dfdb6e (2026-10-13)
export PYTHONPATH=/path/to/cookbook:$PWD/src
export FIREWORKS_API_KEY=...                          # never commit this
# training shape used: accounts/fireworks/trainingShapes/qwen3-4b-minimum (full-parameter, 1x B200)
```

## Data: the pools

**N50000 pool** (`outputs/qwen3-4b/zero-shot-v3/training_pool/train_r3_n50000_seed0_excl/`). 50,000 `train_r3`
examples, 4 samples each from the 4B (temperature 0.6, top-p 0.95, 4096 tokens). Round-1 pass counts:
0/4: 5,254, 1/4: 1,694, 2/4: 1,709, 3/4: 2,590, 4/4: 38,753. Prompts with 1-3 of 4 correct ("mixed") go straight in
(5,993); a small quota of all-correct and all-wrong prompts gets 4 more samples. Pool size 6,196
(`pool.jsonl`); `pool_mixed.jsonl` keeps only the 5,993 mixed rows.

**Teacher verification.** The 397B teacher answered the whole mixed pool (2 votes per prompt). A prompt is
`agree` when every vote equals gold, `split` when some do, `disagree` when none do:

| Gold | agree | split | disagree |
|---|---|---|---|
| entailment | 984 | 129 | 364 |
| neutral | 1,706 | 242 | 590 |
| contradiction | 1,547 | 119 | 312 |
| total | 4,237 | 490 | 1,266 |

The teacher itself agrees with gold on about 74% of this mixed pool (contradiction 80%, neutral 72%, entailment 70%),
and on only about half of the prompts the 4B got 0 of 4 on (0.50 entailment, 0.51 contradiction, 0.25 neutral; small samples). Its disagreements are patterned, not random: gold
contradiction -> teacher "neutral" 85% of the time, gold entailment -> "neutral" 73%, gold neutral ->
"contradiction" 56% / "entailment" 44%. On 48% of the disagreement prompts the 4B's own majority answer equals the
teacher's, so both models reject gold. Some of these are label noise; some may be annotator convention the test
labels share, so dropping them is a bet (not tested separately).

**`scripts/build_clean_pool.py`** writes, next to the input pool:

- `pool_clean.jsonl`: `agree` rows only, balanced to the smallest class: 984 per class, 2,952 rows.
- `pool_clean_unbalanced.jsonl`: all 4,237 `agree` rows.
- `pool_v2.jsonl` (`--neutral-share 0.36 --hard`): see [The last run](#the-last-run-v2-implementation-details). The hard prompts are now searched for in prompt files
  read in sorted order, so the result does not depend on the directory listing. The existing `pool_v2` was built in listing
  order, which is not sorted order and changes the final shuffle, so that order is frozen in
  `reproducibility/pools/hard_candidates.compact.jsonl`; `--hard-candidates` rebuilds the pool from it exactly
  (`tests/test_pools.py` does this from the committed files).

Scripts take their input and output locations as flags; the defaults are under `outputs/`, or `$ANLI_OUTPUT_DIR`
(`anli_classification/paths.py`). The selection logic lives in `anli_classification/pools.py` and `teacher_data.py`
(tested); the scripts only read and write files.
- `clean_summary.json`: the counts above.

## How the RL loop works here

- **Loop:** cookbook `rl_loop`: each optimizer step samples `completions_per_prompt` answers for
  `prompt_groups_per_step` prompts, scores them, and trains on group-relative advantages. Groups whose rewards are
  all equal carry no signal and are dropped (`has_reward_variance`); they still use up rows. One epoch maximum:
  prompts are never repeated (`--epochs 1`; rows are shuffled with seed 0).
- **Base reward:** 1.0 if the parsed answer equals gold, else 0.0. The parser is strict: after any thinking, exactly
  one label word, case-insensitive. Malformed or empty answers score 0.
- **Advantages:** the cookbook z-scores rewards inside each group: `(r - mean) / std`. Consequences that shaped the
  design: (1) anything constant within a prompt's group cancels (class prior, difficulty, a class bonus), so class
  balance has to come from the pool, not the reward; (2) a small reward term that is the only difference inside a group is
  stretched to full size (see the std floor below).
- **Dev eval during training:** a fixed random subset of `dev_r3` (seed 0), one greedy sample per row, at step 0 and every
  `--eval-interval` steps (the stop hook runs after that step's checkpoint is saved).
- **Cost and timing (observed; the dollar rate is an estimate):** at 8 x 64 a step took about 119 s, at 16 x 48 about
  130-150 s. Weight sync is a fixed 36-46 s per step (full-parameter, FILE path, no RDMA). A resumable checkpoint
  takes about 55-65 s. One 16-step run took about 45 minutes and about $20 (reported cost); that works out to roughly
  $0.44 a minute for trainer plus deployment, which is how the other costs here were estimated.

## The last run (v2): implementation details

Run `anli-rl-logs/n50000-v2/`, 12 steps, about 50 minutes. Launch command (key supplied through the environment):

```bash
anli-rl-train \
  --pool-path outputs/qwen3-4b/zero-shot-v3/training_pool/train_r3_n50000_seed0_excl/pool_v2.jsonl \
  --training-shape accounts/fireworks/trainingShapes/qwen3-4b-minimum \
  --completions-per-prompt 16 --prompt-groups-per-step 48 --learning-rate 4e-6 \
  --max-steps 12 --checkpoint-interval 2 --eval-interval 2 --dev-rows 1000 \
  --stop-filter-ratio 0.6 --stop-class-drop 0.10 --class-drop-alert 0.05 \
  --consistency-penalty 0.25 --advantage-std-floor 0.2 \
  --weight-sync-timeout 1800 --stall-timeout 600 --skip-cleanup \
  --log-path ./anli-rl-logs/n50000-v2
```

The same run is one flag now: `anli-rl-train --recipe v2 --pool-path ... --training-shape ... --log-path ...`. Recipes
(`rl/recipes.py`) are named sets of flags used as defaults, explicit flags still win, and `legacy` (the default) changes
nothing. `tests/test_recipes.py` checks `v2` against the run's recorded `run_config.json`.

Defaults that applied: temperature 1.0, 4096 completion tokens, full-parameter (`--lora-rank 0`), eval temperature
0.0, sampling timeout 300 s, delete-on-failure on, promotable intermediate checkpoints on.

**1. The pool (`pool_v2.jsonl`, 3,937 rows).** Built with `build_clean_pool.py --neutral-share 0.36 --hard`:

- Start from the `agree` rows of the mixed pool, balanced: 984 entailment and 984 contradiction.
- Add 297 "trusted hard" prompts: the 4B got 0 of 4 and every teacher vote agrees with gold, from the teacher
  prompt files and not already in the mixed pool (275 entailment, 19 contradiction, 3 neutral).
- Top contradiction up to the entailment count (1,259 each), since contradiction was the weakest dev class.
- Raise neutral to 36% of the pool: 1,419 rows. Final: 1,259 entailment / 1,419 neutral / 1,259 contradiction.

Why the tilt: a balanced pool collapsed neutral within five steps (see the experiment log). The 4B's dominant error on
entailment and contradiction prompts is answering "neutral" (31% and 42% of those samples), so a balanced pool punishes
"neutral" on two thirds of the prompts and rewards it on one third. In a replay of the first five steps through the
cookbook's advantage formula, "neutral" answers had mean advantage -0.167 (entailment +0.059).

**2. The reward (training rollouts only).**

```
R = label_reward - 0.25 * 1[judge says the reasoning concludes a different label than the answer]
```

| Rollout | R |
|---|---|
| correct, reasoning consistent | 1.00 |
| correct, reasoning concludes another label | 0.75 |
| wrong, consistent | 0.00 |
| wrong, reasoning concludes another label | -0.25 |
| malformed or no thinking text | 0.00 (no judge call) |

**Reward v3 (`--reward-version v3`, opt-in; v2 stays the default so the command above is unchanged).** Two gaps in the
v2 reward were found in review: a malformed answer (no judge call) scored 0, more than a wrong answer whose reasoning
contradicts its label (-0.25), and an answer with no reasoning text skipped the judge, so it could never be penalised. v3
(with a penalty above 0) scores a malformed answer `-(penalty + 0.25)`, below every parseable outcome, and penalises a
parseable answer with no reasoning like an inconsistent one (verdict `no_reasoning`). The v2 logs show no sign the gaps were
exploited (malformed fell from about 1.5% to 0%, no-reasoning stayed near 0%), so they were a latent incentive, not an
observed one. Runs without a penalty are unaffected by the version.

**v3 has not been used in a training run.** It has unit tests only, so its effect on results is unmeasured. v2 is the default
in the CLI, in `make_rollout_fn`, in `training_reward` and in the `v2` recipe, and `tests/test_reward_default.py` pins that,
together with the exact v2 reward values, so the v2 run stays reproducible.

Evaluation and the `outcome` field in `rollouts.jsonl` use the strict label reward only; the shaped reward is logged as
`reward`, the strict one as `label_reward`, the verdict as `judge_verdict`.

**3. The consistency judge (`ConsistencyJudge` in `rl_train.py`).**

- Model: `accounts/fireworks/models/gpt-oss-120b` on serverless (`reasoning_effort` low, `max_tokens` 300, temperature 0).
  Two other serverless models (`deepseek-v4p1-flash`, `glm-5p3-flash`) returned nothing parseable in a test, probably
  because their reasoning used the token limit; they were not pursued.
- Input: the premise, the hypothesis, the model's full reasoning (if longer than 16,000 characters: first 3,000 plus
  last 12,000) and the label the model output. It is **not** given the gold label, so it cannot become a correctness
  grader. Reply format `<verdict> <label>`; only the verdict (`consistent`, `inconsistent`, `unclear`) is used, because
  on one test case the label field contradicted its own verdict.
- Fails open: an error, timeout or unparseable reply gives no verdict (so no penalty); three attempts with backoff; replies
  are cached by input. A judge outage cannot stop or distort training.
- Validation before use: on 120 real rollouts it flagged 4 (3.3%), all genuine (reasoning ends "the answer is neutral",
  output `contradiction`), and agreed with a rule-based matcher on all 112 cases where the rule fired.
- In the run: 13,930 calls, 123 inconsistent (0.9% overall), 6 unclear, 0 errors, 0 unparsed. The flagged rate was
  about 3% for the first 2,500 calls and under 1.5% afterwards. Cost was not priced (about 800 prompt tokens per call).

**4. Std floor on advantages (`--advantage-std-floor 0.2`).** Passed to the cookbook as its `advantage_fn`
(`floored_advantages`). Without it, one penalised sample among 16 otherwise identical correct answers would get
advantage `(0.75 - 0.984) / 0.0625 = -3.75` (the same size as a lone success among failures), whatever the penalty is.
With the floor it gets about -1.17 and scales with the penalty. Ordinary binary groups (std at least 0.25) are
unchanged; equal-reward groups stay at zero advantage.

**5. Per-class dev metrics and guards.**

- Each eval logs `dev/recall_*`, `dev/predicted_share_*` and `dev/macro_f1` (failed rollouts count as wrong and as no
  prediction). Dev rows default to 1,000 (about 330 per class).
- `--class-drop-alert` prints an ALERT when a class's dev recall is more than 0.05 below the run's first eval;
  `--stop-class-drop` stops the run at the eval where the drop exceeds the limit (0.10 here).
- `--stop-filter-ratio` stops when the mean filter ratio of the last 3 steps exceeds the limit (0.6 here; 0.5 earlier, raised
  because the hard prompts raise the baseline). Note that with the judge penalty the filter ratio means something
  slightly different: an all-correct group with one flagged sample now has variance and counts as usable.
- `--stall-timeout` (600 s here) kills the run when neither `rollouts.jsonl` nor `metrics.jsonl` changes, and deletes the
  trainer and deployment it created. `--weight-sync-timeout` (default 1800 s) is how long the cookbook waits for a
  hot-loaded snapshot. Delete-on-failure also runs on a crash, and finds a trainer that was created but never wrote
  `trainer_job.json` (still pending) by creation time and base model.
- Every run ends by writing `run_result.json` (`status`: completed, stopped, failed or stalled; `stop_reason`:
  max_steps, filter_ratio or class_drop; the final step; job ids; recipe; git commit), so tools need not parse the log.
  `run_config.json` also records the git commit and whether the tree was dirty.
- Cleanup deletes the recorded trainer and `--deployment-id` for certain. A trainer or deployment the run created but never
  recorded can only be found by creation time (within 60 s before the run started or later, on the base model). That guess
  can match someone else's resource on a shared account, so each match is logged as "matched by creation time" and
  `--no-cleanup-heuristic` turns it off.
- The cookbook has no stop hook, so a guard ends a run by raising `StopAtStep`, a `BaseException`, from the evaluation
  function (the cookbook swallows ordinary exceptions there). `tests/test_cookbook_contract.py` checks this against the
  installed cookbook; run it where the cookbook is importable.
- With `--skip-cleanup` the trainer and deployment stay up after the run (so a run can be resumed or checkpoints
  promoted) and keep billing until deleted.

**6. Result and what was kept.** Dev (1,000 rows, greedy), checkpoint and eval every 2 steps:

| Step | Accuracy | Macro-F1 | Recall (entail / neutral / contra) | Share of predictions (E / N / C) |
|---|---|---|---|---|
| 0 | 0.656 | 0.658 | 0.73 / 0.70 / 0.53 | 33 / 42 / 24 |
| 2 | 0.646 | 0.650 | 0.68 / 0.72 / 0.53 | 30 / 45 / 24 |
| 4 | 0.659 | 0.662 | 0.65 / 0.74 / 0.58 | 28 / 45 / 27 |
| 6 | 0.670 | 0.674 | 0.65 / 0.69 / 0.68 | 27 / 41 / 32 |
| 8 | 0.682 | 0.685 | 0.69 / 0.67 / 0.69 | 28 / 39 / 33 |
| 10 | 0.692 | 0.693 | 0.71 / 0.63 / 0.74 | 30 / 34 / 35 |
| 12 | 0.694 | 0.694 | 0.75 / 0.59 / 0.74 | 32 / 32 / 36 |

The run stopped at its 12-step cap, and the neutral guard also tripped there (neutral recall 0.107 below baseline).
Step 10 was promoted: `anli-grpo-v2-step10` (`scripts/promote_checkpoint.py --checkpoint step-10-<id>`), then the
trainer and deployment were deleted. Training reward rose from 0.55 (step 1) to about 0.68, and filter ratio from 0.11 to
about 0.46; `ppo_kl` stayed at 0.0009-0.0014.
