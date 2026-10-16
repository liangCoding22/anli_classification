# Reproducing the reported results

Everything here was exported by `scripts/export_repro_artifacts.py` from the runs described in the top-level README
("Training (GRPO) and experiment log"). It holds what is small enough to share; the large inputs stay out of git.

## What is here

```
reproducibility/
  manifest.json            sha256 and row counts of every large local file, plus the environment (Python, package
                           versions, repo commit, cookbook commit, training shape version)
  pools/                   compact pools: key, gold, premise, hypothesis, reason, group, 4B pass counts, teacher votes
    pool_mixed.compact.jsonl   5,993 mixed prompts (4B got 1-3 of 4 right)
    pool_clean.compact.jsonl   2,952: teacher-verified, balanced 984 per class
    pool_v2.compact.jsonl      3,937: the pool of the last run (neutral 36%, 297 trusted hard prompts)
    teacher_votes.jsonl        every parsed teacher vote per prompt (397B teacher)
    clean_summary.json, n50000_summary.json
  runs/<run>/              run_config.json (arguments and the example keys), metrics.jsonl (every metric the cookbook
                           logged, plus evals), stdout.log, trainer_job.json, dataloader.json, and
                           rollouts*.compact.csv.gz (key, gold, pred, reward, label_reward, judge_verdict, ...)
  results/{promoted,base}/ test_r3 predictions (key, gold, pred, error) and metrics.json
```

The compact pool files can be passed straight to `anli-rl-train --pool-path`: they carry every field the loader reads.

## What is not here, and why

| Not included | Size | Why / how to check a local copy |
|---|---|---|
| Full pools (with the 4B's sampled reasoning), `pool.jsonl`, `predictions.jsonl` | 36-475 MB | Too large for git. `manifest.json` has each file's sha256. |
| Teacher traces (397B teacher, reasoning included) | 175 MB | Too large. sha256 in the manifest; every vote is in `pools/teacher_votes.jsonl`. |
| Full rollout logs with completions and thinking | 20-50 MB per run | Compact columns only; the originals' sha256 is in the manifest. |
| The promoted models | | They live in the Fireworks account (`anli-grpo-v2-step10`, `anli-grpo-clean-step5`), not in git. |

## Re-running the last run

Environment (from the manifest): Python 3.12, the cookbook at the recorded commit, training shape version
`qwen3-4b-minimum/versions/i2yojzpo`, judge `accounts/fireworks/models/gpt-oss-120b`. Then the command in the top-level
README ("The last run (v2)") with `--pool-path reproducibility/pools/pool_v2.compact.jsonl`.

A rerun will not match bit for bit: training samples at temperature 1.0 without a seed, the judge and the base model are
hosted, and the Training API is a private preview. Expect the same behavior, not the same weights. The claim worth
checking is the test result, which is statistical (paired +3.8 to +9.0 points, McNemar z about 4.7).

## Checking the reported numbers from the stored predictions

`tests/test_reproducibility.py` recomputes test_r3 accuracy from `results/*/test_r3.compact.jsonl` and checks it against
the reported 0.7008 (promoted) and 0.6367 (base), and checks the pools against the manifest. A change to the metrics code
that moved a reported number would fail it.

## Data license

The prompt text is ANLI (Nie et al.), CC BY-NC 4.0: https://huggingface.co/datasets/facebook/anli. The compact pools and
predictions are derived from it, so they are for non-commercial use with attribution.
