# anli_classification

Fine-tune a transformer for **3-way NLI** on [ANLI](https://github.com/facebookresearch/anli) (entailment / neutral / contradiction).

This repo is a small training codebase, not a research dump: load data, wrap a Hugging Face classifier, train, then score each ANLI test round.

## Architecture

```
src/anli_classification/
  config.py      defaults (model, batch size, rounds)
  data.py        load + concatenate ANLI r1/r2/r3, tokenize premise+hypothesis
  model.py       AutoModelForSequenceClassification (3 labels)
  metrics.py     accuracy + macro-F1
  train.py       Hugging Face Trainer loop  ← main entry
  evaluate.py    score saved checkpoint on test_r1 / test_r2 / test_r3
```

**Data flow:** `facebook/anli` → concat train/dev rounds → tokenize `(premise, hypothesis)` → `Trainer` → checkpoint under `outputs/` → per-round test metrics.

ANLI’s official labels are already `0/1/2` in that order. `LABEL2ID` in `__init__.py` is the source of truth if you remap or add a dataset.

## How to run

You need Python 3.10+ and [Apple Command Line Tools](https://developer.apple.com/download/more/) (`xcode-select --install`) so `git` and compiling wheels work.

```bash
cd anli_classification
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

anli-train --epochs 1
anli-eval --checkpoint outputs/anli-distilbert
pytest
```

`distilbert-base-uncased` is the default so a first GPU/CPU run stays cheap. Swap models with `--model-name roberta-base`.

## Sync to GitHub

This machine did not have working `git` when the project was created (no Xcode CLT, Homebrew not writable, `gh` missing). After CLT is installed:

```bash
cd anli_classification
git init
git add .
git commit -m "Initial ANLI classification trainer."
gh auth login
gh repo create anli_classification --public --source . --remote origin --push
```

If you prefer the website: create an empty public repo named `anli_classification`, then `git remote add origin git@github.com:<you>/anli_classification.git` and `git push -u origin main`.

## Best first change

Change the default encoder in `src/anli_classification/config.py` (`model_name`) or add a `--train-rounds r1` flag in `train.py` so you can overfit a single round before a full r1–r3 run. Keep label IDs in `__init__.py` unless you are adding a new class.
