# anli_classification

Improving `qwen3-4b` on [ANLI](https://huggingface.co/datasets/facebook/anli) round 3 (3-way NLI; labels `entailment` 0,
`neutral` 1, `contradiction` 2) with GRPO on Fireworks, plus the tooling to evaluate it without fooling yourself.

## Results at a glance

ANLI **test_r3**, 1,200 examples, `zero-shot-v3` prompt, thinking on, greedy, 4096 tokens:

| | Base `qwen3-4b` | `anli-grpo-v2-step10` |
|---|---|---|
| Accuracy | 0.6367 | **0.7008** |
| Macro-F1 | 0.6415 | **0.7012** |
| Recall: entailment / neutral / contradiction | 0.664 / 0.677 / 0.568 | 0.674 / 0.629 / **0.801** |

Paired on the same examples: +6.4 points, 95% interval +3.8 to +9.0, McNemar z = 4.7. On dev the run went 0.656 -> 0.694
in 12 steps. Most of the gain is a better-calibrated "neutral" boundary and a large contradiction gain; entailment-versus-
contradiction skill barely moved, and neutral recall paid for it (0.70 -> 0.63). Details, the run log and the other
experiments (including what did not work) are in [docs/experiments.md](docs/experiments.md).

## The recipe in five points

1. **Pool:** prompts the 4B gets right 1-3 of 4 times, kept only when a 397B teacher agrees with gold, entailment and
   contradiction balanced, **neutral at 36%** (a balanced pool made neutral collapse in five steps), plus 297 prompts the 4B
   always misses and the teacher gets right.
2. **Reward:** 1 for the gold label, minus 0.25 when an LLM judge finds the reasoning concluding a different label than the
   answer. The judge sees premise, hypothesis, reasoning and the answer, never gold.
3. **Advantages:** GRPO's per-group z-score with a std floor of 0.2, so a lone small penalty keeps its size.
4. **Gentle optimisation with guards:** 16 completions x 48 prompts, learning rate 4e-6, 12 steps, checkpoint and a
   1,000-row dev eval every 2 steps, and stops on a class's recall collapsing or groups going uniform.
5. **Choose by balance, not the last step:** pick the checkpoint with the best dev macro-F1 and no collapsed class
   (step 10), promote it, evaluate it with `--template client`.

Implementation details: [docs/training.md](docs/training.md). The whole recipe is one flag, `--recipe v2`.

## Quickstart

```bash
# evaluate (chat backend; see docs/evaluation.md for batch, pass@k and result folders)
anli-eval --splits dev --rounds r3 --limit 200 --temperature 0.0

# train the last run's recipe (needs the Fireworks cookbook, Python 3.11+, the Training API; see docs/training.md)
anli-rl-train --recipe v2 --pool-path reproducibility/pools/pool_v2.compact.jsonl \
  --training-shape accounts/fireworks/trainingShapes/qwen3-4b-minimum --log-path ./anli-rl-logs/my-run

# promote a checkpoint, then evaluate it on a short-lived deployment (deleted afterwards)
python scripts/promote_checkpoint.py --job-id <trainer-job> --list
python scripts/promote_checkpoint.py --job-id <trainer-job> --checkpoint <step-N-id> --output-model-id <model-id>
python scripts/eval_promoted.py --promoted-model accounts/<account>/models/<model-id> --deployment-id <model-id>-eval \
  --splits test --rounds r3 --temperature 0.0
```

Rebuilding the pools and checking the reported numbers from stored files: [reproducibility/README.md](reproducibility/README.md).

## Eval caveats (read before trusting a number)

1. **Cached predictions are keyed by example, so settings matter.** Each results folder holds a `run_spec.json` (model,
   prompt, template, temperature, top-p, top-k, max tokens, samples). A run with different settings is refused instead of
   silently reusing the old answers; `--run-tag NAME` writes separate results. Folders made before this have no spec: they
   are reused with a warning and their settings are unknown.
2. **The default is not greedy.** `anli-eval` samples at temperature 0.6, top-p 0.95. Every number in this README is greedy
   (`--temperature 0.0`). Compare only runs with the same settings.
3. **Promoted RL checkpoints need `--template client`.** Their server template switches thinking off, and scoring them
   without it measured a different model (dev_r3 0.42 against 0.72 in-loop). See [docs/evaluation.md](docs/evaluation.md).
4. **A reply cut off while thinking counts as wrong.** The limit is `--max-tokens` (4096). Check `unparsed` and
   `with_reasoning` in `metrics.json`: the base model left 11 of 1,200 unparsed on test_r3.
5. **Results belong to the model, not the deployment.** For `<model>#<deployment>` the results folder is the model's.
   Folders made earlier under a deployment id are not found automatically.
6. **A deployment answers a little after it says READY.** Requests sent right away can fail with 404 (8 of 1,200 did).
   `eval_promoted.py` now waits for a test request to succeed; re-running the same command retries only failed rows.
7. **Dev is for choosing, test is read once for reporting.** The 1,000-row dev subset (seed 0) has a standard error of about
   1.5 points on accuracy and 2.6 per class. Compare models as paired differences on the same examples, and do not read
   more into a one-point move.
8. **Greedy is not bit-exact on a hosted deployment.** The same checkpoint scored 0.652 and 0.668 on the same 500 dev rows
   in two evaluations. Expect a few rows of drift.
9. **Prompt versions never mix.** Each `--prompt-version` writes to its own folder; do not edit an existing prompt's wording.
   `zero-shot-v4a` and `zero-shot-v4b` are experiments and both scored below v3 (see experiments).
10. **The batch API cannot use the client template** (it takes messages only), so it cannot score promoted RL models. It is
    about half the price but slower.

## Documentation

- [docs/evaluation.md](docs/evaluation.md): `anli-eval`, batch and pass@k evals, promoted checkpoints, result folders.
- [docs/training.md](docs/training.md): the pipeline, pools and teacher verification, the RL loop, the last run's implementation.
- [docs/experiments.md](docs/experiments.md): test results, the run log, what we learned, prompt experiments.
- [docs/operations.md](docs/operations.md): capacity and network failures, cleanup, what is not verified.
- [reproducibility/README.md](reproducibility/README.md): compact pools, run records, predictions and a manifest.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export FIREWORKS_API_KEY=...   # never commit this
```

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
- `run_spec.py`: what a results folder was produced with, so a cache is never reused for other settings.
  `deploy.py`: waits until a new deployment answers requests.
- `rl/`: the GRPO driver, split by concern: `reward` (label reward, LLM consistency judge, advantages; no cookbook imports),
  `guards`, `stop`, `cookbook_patches`, `cleanup`, `launch`, `cli`, `recipes`. `rl_train.py` re-exports it, so the
  `anli-rl-train` entry point and old imports are unchanged. `rl_summary.py`, `rl_checkpoints.py`: run summary, checkpoint
  listing and promotion.
- `pools.py`, `teacher_data.py`, `paths.py`: teacher-verified pools, teacher prompts/traces/SFT filters, default locations.
  `training_pool.py`, `sft_data.py`, `sft_train.py`: pool sampling and the earlier SFT pipeline.

## Tests

```bash
pytest   # offline; uses a fake backend, no API calls
```

The batch backend's HTTP calls are not covered by live tests yet. Run a small
batch job before a full one.
