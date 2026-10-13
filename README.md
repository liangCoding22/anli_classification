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

- **Model slug:** the last path segment of the model name, e.g. `nemotron-lightning-3p5-30b-a3b`.
- **Prompt version:** `NliPrompt.VERSION` in `prompts.py` (currently `zero-shot-v1`).
  Bump it whenever the wording changes, so results from different prompts never mix.
  Model changes get a new slug directory automatically.
- **Resuming:** runs read the slice JSONL files first. Successful predictions are kept,
  and failed ones (errors, empty replies) are retried.
- **Caveat:** `anli-batch-report` files results under the *current* prompt version. If the
  job used an older prompt, check the job's `input.jsonl` before trusting that directory.

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

## Tests

```bash
pytest   # offline; uses a fake backend, no API calls
```

The batch backend's HTTP calls are not covered by live tests yet. Run a small
batch job before a full one.
