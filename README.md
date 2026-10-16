# anli_classification

Prompt-based NLI classification on [ANLI](https://huggingface.co/datasets/facebook/anli)
through the Fireworks API. Labels: `entailment` (0), `neutral` (1), `contradiction` (2).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export FIREWORKS_API_KEY=...   # never commit this
```

## Functionality

### 1. Smoke-test inference and eval (`anli-eval`, chat backend)

Concurrent chat-completion calls, for quick checks and debugging.

```bash
anli-eval --limit 10                       # 10 random examples per split/round
anli-eval --splits dev --rounds r1 --limit 50 --concurrency 16
anli-eval --model accounts/fireworks/models/<name> --max-tokens 4096
```

Each slice is `<split>_<round>` (e.g. `dev_r1`). Splits default to `dev test`, rounds to `r1 r2 r3`.
`--limit N` draws a sample with `--seed` (default 0), so the same flags give the same examples.

### 2. Batch eval (`anli-eval --backend batch`)

Submits each chunk of `--batch-size` examples (default 256) as one Fireworks batch
inference job, polls until it finishes, and parses the output rows. Roughly half the
serverless per-token price, but slower. Requires `--account-id`.

```bash
anli-eval --backend batch --account-id <account> --batch-size 1000
```

### 3. Post-processing a finished batch job (`anli-batch-report`)

Turns a completed job into the same outputs as `anli-eval`, so batch and smoke runs are
directly comparable. Use it to re-score a job, or to score one submitted elsewhere.

```bash
anli-batch-report --job-id anli-job-6b9329a3a3 --account-id <account>
```

The model comes from the job record. Downloaded artifacts are reused on re-runs.
Rows the batch returned with an error, or not at all, count as unparsed errors in the metrics.

### 4. pass@1 and pass@3 (`--samples 3`)

Each example is answered `--samples` times. Samples use `--temperature` (default 0.6), so repeats can differ. Metrics then include:

- `pass@1`: the average share of correct samples per example.
- `pass@3`: the share of examples where at least one of the samples is correct.

```bash
anli-eval --samples 3 --splits dev --rounds r3 --limit 1000
```

Sample 0 keeps the plain example key; extra samples are keyed `<key>#1`, `<key>#2`.
Output files are unchanged, and each sample is a line in the slice's JSONL.

### 5. Evaluating promoted RL checkpoints (`--template client`)

A promoted RL checkpoint is not served with the base model's chat template: Fireworks registers
it with `useHfApplyChatTemplate: false`, and its built-in qwen3 template prefills an empty
`<think></think>`, which turns thinking off. RL trained and evaluated with thinking on, so a
plain `anli-eval` of the promoted model scores a different model (dev_r3: 0.42 vs 0.72 in-loop).

`--template client` renders the prompt locally with the HF tokenizer's chat template (thinking
on, the same tokens as training) and sends it to `/completions`, so no server template applies.
Results go to `<prompt-version>-client-template/`. It needs the chat backend against a deployment
of the model. The batch API cannot do it: it takes only `messages` (`prompt_token_ids` rows fail
with HTTP 400), and for a promoted model it ignores `chat_template_kwargs` and rejects
`reasoning_effort` ("non-reasoning model").

```bash
pip install -e ".[template]"
anli-eval --model accounts/<account>/models/<promoted-id> \
    --splits dev --rounds r3 --limit 200 --template client --tokenizer-model Qwen/Qwen3-4B
```

`anli-eval` and `anli-batch-report` print a warning when no reply in a slice has reasoning;
`metrics.json` counts these replies as `with_reasoning`.

### Outputs and versioning

Everything for one model and one prompt version lands in one directory:

```
outputs/<model-slug>/<prompt-version>/
├── dev_r1.jsonl ...        one line per example: key, gold, pred, raw, reasoning, error
├── metrics.json            per slice and "overall": n, accuracy, macro_f1, per-class
│                           precision/recall/F1, confusion matrix, unparsed, errors
└── batch_jobs/<job-id>/    batch runs and reports only
    ├── input.jsonl         requests sent (when the API allows downloading them)
    ├── output.jsonl        raw response rows
    ├── job.json            Fireworks job record
    ├── metrics.json        same shape as the run's metrics.json
    └── summary.json        job metadata, row counts, headline metrics
```

- **Model slug:** the last path segment of the model name, e.g. `qwen3-4b`. For `<model>#<deployment>` (a promoted
  model on a deployment) it is the model's, not the deployment's; results made before this fix sit in folders named
  after the deployment id and are not found automatically.
- **Prompt version:** the `VERSION` of a prompt class in `prompts.py`. Available: `zero-shot-v3`
  (default, `NliPromptV3`, defines the labels as logical relations), `zero-shot-v2`
  (`NliPrompt`), and two experimental variants, `zero-shot-v4a` and `zero-shot-v4b`, that both scored
  below v3 (see "Prompt experiments"). Pick one with `--prompt-version`; each version writes to its own directory, so
  results from different prompts never mix. Add a new class and register it in `PROMPTS` for
  further variants; don't edit an existing prompt's wording.
  Model changes get a new slug directory automatically.
- **Resuming:** runs read the slice JSONL files first. Successful predictions are kept,
  and failed requests (API errors, rows missing from a batch) are retried. An empty reply
  (thinking used up `--max-tokens`) is an answer and counts as wrong, as in RL; it is not retried.
- **Run settings:** each folder holds a `run_spec.json` with the settings that produced its cached predictions (model,
  prompt version, template, temperature, top-p, top-k, max tokens, samples). A run with different settings is refused,
  because it would otherwise find every example cached and report the old answers; pass `--run-tag NAME` to write it to
  `<prompt-version>--NAME`. A folder with predictions but no spec (made before this) is reused with a warning.
- **Caveat:** `anli-batch-report` files results under the *current* prompt version. If the
  job used an older prompt, check the job's `input.jsonl` before trusting that directory.

## Training (GRPO) and experiment log

Everything below is about improving `qwen3-4b` on ANLI r3 with GRPO, using the Fireworks cookbook's
strictly on-policy loop. Numbers are from the logs under `anli-rl-logs/` and `outputs/`; where something is
an estimate or was not verified, it says so.

### Pipeline components

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

### Data: the pools

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
- `pool_v2.jsonl` (`--neutral-share 0.36 --hard`): see "The last run".
- `clean_summary.json`: the counts above.

### How the RL loop works here

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

### The last run (v2): implementation details

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

### Evaluation: test_r3 (v3 prompt, thinking on, greedy, 4096 tokens, 1,200 examples)

Promoted model: `scripts/eval_promoted.py --promoted-model accounts/<acct>/models/anli-grpo-v2-step10
--deployment-id <id> --prompt-version zero-shot-v3 --splits test --rounds r3 --temperature 0.0` (client template).
Base model: `anli-eval --backend batch --model accounts/fireworks/models/qwen3-4b --splits test --rounds r3
--temperature 0.0 --max-tokens 4096 --batch-size 1200` (server template, all 1,200 replies had reasoning).

| | Base `qwen3-4b` | `anli-grpo-v2-step10` |
|---|---|---|
| Accuracy | 0.6367 | **0.7008** |
| Macro-F1 | 0.6415 | **0.7012** |
| Recall: entailment / neutral / contradiction | 0.664 / 0.677 / 0.568 | 0.674 / 0.629 / **0.801** |
| Share of predictions: E / N / C | 30 / 42 / 27 | 28 / 33 / 39 |
| Unparsed / errors | 11 / 0 | 1 / 1 |

Paired on the same examples: 167 only the promoted model got right, 90 only the base got right, net +77 (+6.4 points),
McNemar z = 4.7, paired bootstrap 95% interval +3.8 to +9.0 points. The in-training dev numbers for step 10 (0.692 /
0.693) are close to the test result, so the checkpoint choice does not look overfit to dev. Test round r1/r2 were not evaluated.

Gotchas for this eval: requests sent right after a deployment reports READY can fail with 404 "model not found" (8 of
1,200 did; re-running the same command retries only failed rows; the results folder is named after the deployment id, so
copy it when using a new deployment id). A deployment can also end FAILED with no reason in the API; one did, and a retry
on a fresh deployment id worked.

### Experiment log

| Run (log dir) | Setup | Outcome |
|---|---|---|
| `union-1epoch-lr3e-6-hotload900` | 1,493-row pool, 8 x 64, lr 3e-6, 16 steps | Training reward 0.58 -> 0.59; filter ratio rose from about 0.2 to 0.55 (the "groups go uniform" effect); about 45 min, about $20. |
| `n50000-pilot3` | `pool_mixed`, 16 x 48, lr 3e-6 for steps 1-10, then resumed from the step-10 checkpoint at 6e-6 | Dev (500 rows, greedy) flat: 0.658, 0.658, 0.652 / 0.668 (step 10), 0.668 (step 15); `ppo_kl` 0.001 -> 0.0017. Stopped at step 17 when the Mac's network dropped; resources deleted. |
| `n50000-clean3` | `pool_clean` (class-balanced, teacher-verified), 16 x 48, lr 6e-6 | Neutral collapsed in about 5 steps (neutral share of training answers 35% -> 3%, neutral reward 0.61 -> 0.08). Dev 0.654 -> 0.683 accuracy, macro-F1 0.657 -> 0.673, recall 0.73/0.70/0.53 -> 0.88/0.46/0.71. Stopped at step 5 by the filter-ratio guard (0.34, 0.66, 0.69 at steps 3-5). Promoted as `anli-grpo-clean-step5`. |
| `n50000-v2` | `pool_v2`, judge penalty, std floor, class guards, lr 4e-6, 12 steps | Dev 0.656 -> 0.694; test_r3 0.7008 vs base 0.6367 (above). Promoted step 10: `anli-grpo-v2-step10`. |

Failed or abandoned launches (capacity, network) are not listed; see "Operational notes".

### What we learned

1. **Plain GRPO on this pool barely moves dev accuracy.** At 3e-6 and 6e-6 `ppo_kl` stayed near 0.001-0.0017 and dev
   stayed flat. Training reward rose (0.58 -> 0.65) because sampled answers at temperature 1.0 became more consistent
   (the share of all-correct groups roughly tripled, all-wrong stayed near 4%); greedy dev accuracy, which already picks
   the mode, did not change.
2. **Gains are mostly class-prior movement at the neutral boundary.** Row-level comparison of the same 1,000 dev rows:
   - Balanced-pool run, step 0 -> 5: net +29 rows; neutral boundary about +21, entailment<->contradiction flips 0
     (11 fixed, 11 broken), format about +7.
   - v2 run, step 0 -> 12: net +38 rows; neutral boundary +28 (+105 fixed, -77 broken), entailment<->contradiction
     flips +6 (+14 / -8, within noise), format about +4.
   The base model over-answers neutral (42% of predictions vs 34% gold); moving that rate toward the gold share is worth
   roughly +2 to +3 points. Going past it costs neutral recall.
3. **Class mix is a strong lever, and a balanced pool is not neutral.** Entailment and contradiction errors mostly land
   on "neutral", so a 33/33/33 pool pushed neutral down hard, while the original 25/42/33 mix pushed it up. A neutral
   share of about 36% with a lower rate (4e-6) and checkpoints every 2 steps kept all classes at 0.59 or better.
4. **The consistency penalty worked on its target.** Judge-flagged inconsistency (reasoning ends on label A, answer is
   B) fell from about 3% to under 1.5% in 3-4 steps, and invalid answers fell from 0.6% to about 0.1-0.2%. In the
   flagged cases the reasoning was often right (in the 4-case test sample, 3 ended on the gold label: "so the answer is
   neutral", output `contradiction`). The invalid-answer drop is on dev.
5. **Label noise caps what this pool can teach.** The 397B teacher agrees with gold on only about 74% of the mixed pool;
   for 22 of the 57 prompts the 4B got wrong in all 16 samples (39%), every teacher vote also rejected gold. Cleaning the pool by teacher agreement
   changed starting difficulty little (predicted start reward 0.63 vs 0.62). It also removes the prompts the teacher
   rejects, which supplied about 24% of the usable (nonzero-gradient) groups early in the run.
6. **What the promoted model learned (examples, test_r3).** Real reading gains: noticing "She" refers to Val
   (hypothesis "both male" -> contradiction), and separating "played a lady's maid" from "worked as one". Heuristic gains
   with thin reasoning: "important to California's independence" -> contradiction "because it's not mentioned".
   Regressions on gold-neutral items where it now calls an unsupported specific claim ("twelve times", "only practiced in
   3 countries") a contradiction. Overall it adopted a "specific claim the premise doesn't support -> contradiction" habit,
   which explains the contradiction gain (+23 points) and the neutral loss (-5).
7. **Cost control needs automation.** Runs on a laptop stalled on network drops while the GPUs kept billing; the stall
   watchdog and delete-on-failure exist because of that.

### Prompt experiments (base model, dev_r3, 1,000 rows seed 0, thinking on, greedy)

Question: can a prompt fix the base model's weak contradiction recall (0.538, 120 of 325 contradictions answered
"neutral")? Two variants added to `prompts.py`: `zero-shot-v4a` (wider contradiction definition, stricter neutral, "do
not choose neutral because the premise does not mention the claim") and `zero-shot-v4b` (ordered questions:
entailment, then contradiction, then neutral as the remainder). They were tuned on dev, not test.

| Prompt | Accuracy | Macro-F1 | Recall (E / N / C) | Neutral share | Unparsed |
|---|---|---|---|---|---|
| v3 (baseline) | 0.655 | 0.658 | 0.715 / 0.707 / 0.538 | 43% | 7 |
| v4a | 0.633 | 0.638 | 0.653 / 0.743 / 0.498 | 47% | 15 |
| v4b | 0.627 | 0.625 | 0.662 / 0.769 / 0.443 | 51% | 2 |

Both are worse and moved the wrong way: contradiction recall fell and neutral predictions rose (net -22 and -28 rows of
1,000 vs v3; McNemar z 1.8 and 2.1). My guess, not verified: defining neutral as "compatible but not established" gave the
model an easier reading of neutral than the stricter contradiction wording did. Only two variants, one run each. The RL
result (contradiction recall 0.568 -> 0.801 on test) is far larger than anything these prompts did.

### Operational notes

- **Capacity:** two launches in a row had the trainer stuck `PENDING` (36 and 10 minutes) with no reason given; the
  stall watchdog cleaned up each time. Later launches got a GPU within minutes.
- **Network:** run training from a stable connection with the machine awake (`caffeinate`). Connection errors made the
  sampler retry, and one eval stall (81 of 500 requests in 45 minutes) was on a connection that looked like a phone hotspot.
- **Promoted models:** evaluate them with `--template client` (section 5); the server template disables thinking.
- **Deleting:** `scripts/promote_checkpoint.py` first, then delete the trainer; promoted models survive deleting the
  trainer and deployment (verified for both promoted checkpoints).
- **Unverified or not done:** the cookbook's exact advantage call path in the synchronous loop was inferred from a single
  call site; the judge's serverless price was not checked; no teacher-disagreement ablation (keeping those prompts); the
  v4 prompts were not tried on the promoted model; r1 and r2 test rounds were not run. The full pools and teacher traces
  stay out of git (`outputs/` is gitignored); compact pools, run configs and logs, and the test predictions are in
  `reproducibility/` (see its README).

## Architecture

```
AnliDataset ─► Example ─► NliClassifier ─► Prediction ─► Evaluator ─► metrics.json
                           │                               │
                           ├─ NliPrompt (builds messages)  ├─ PredictionStore (JSONL cache)
                           ├─ ChatBackend (LLM call)       └─ MetricsCalculator
                           └─ LabelParser (reply → label)
```

- `data.py`: `AnliDataset` loads splits; `limit` draws a seeded random sample.
- `prompts.py`: `NliPrompt` (versioned). `parsers.py`: `LabelParser`.
- `llm/base.py`: `ChatBackend` interface. `llm/fireworks.py`:
  `FireworksChatBackend` (concurrent calls) and `FireworksBatchBackend` (batch API).
- `classifier.py`: `NliClassifier` ties prompt, backend, and parser together.
- `evaluator.py`: runs the classifier over slices in batches, using the store to resume.
- `config.py`: `RunConfig`. `cli.py`: builds the components and runs them.
- `rl_train.py`: GRPO driver (reward, `ConsistencyJudge`, `floored_advantages`, guards, per-class dev metrics,
  failure cleanup). `rl_summary.py`, `rl_checkpoints.py`: run summary, checkpoint listing and promotion.
  `training_pool.py`, `sft_data.py`, `sft_train.py`: pool building and the earlier SFT pipeline.

## Tests

```bash
pytest   # offline; uses a fake backend, no API calls
```

The batch backend's HTTP calls are not covered by live tests yet. Run a small
batch job before a full one.
