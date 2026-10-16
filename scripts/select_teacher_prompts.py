"""Pick training prompts for teacher-trace generation, by how hard the 4B found them.

Reads the 4B's round-1 answers (samples 0-3, from `anli-training-pool`) for every run under
`outputs/qwen3-4b/zero-shot-v3/training_pool/`, groups each example by how many of the 4 answers
were correct, and writes a prompt file:

- every mixed example (1-3 of 4 correct),
- always-wrong examples (0 of 4) and always-correct examples (4 of 4), sampled to fill the rest
  and to even out the three labels (the mixed group leans neutral).

    PYTHONPATH=src .venv-sft/bin/python scripts/select_teacher_prompts.py --total 10000

Output: outputs/teacher_prompts/prompts_<total>.jsonl and summary_<total>.json. Runs under
`training_pool/` are disjoint (each excludes the earlier ones) and hold only train_r3 examples, so
nothing here overlaps the dev split.
"""

import argparse
import glob
import json
import random
from collections import Counter
from pathlib import Path

POOL_DIR = "outputs/qwen3-4b/zero-shot-v3/training_pool"
OUT_DIR = Path("outputs/teacher_prompts")
LABELS = ("entailment", "neutral", "contradiction")
ROUND_1 = 4


def load_examples() -> list[dict]:
    """One record per example with a complete, error-free round 1, and its number of correct answers."""
    examples: dict[str, dict] = {}
    for run in sorted(glob.glob(f"{POOL_DIR}/train_r3_n*")):
        samples: dict[str, dict[int, bool]] = {}
        info: dict[str, dict] = {}
        for line in open(f"{run}/predictions.jsonl"):
            r = json.loads(line)
            base, _, idx = r["key"].partition("#")
            idx = int(idx or 0)
            if idx >= ROUND_1:
                continue
            if r.get("error"):  # errored answers are retried later in the file; keep only ok ones
                continue
            samples.setdefault(base, {})[idx] = r["pred"] == r["gold"]
            info[base] = {k: r[k] for k in ("gold", "premise", "hypothesis", "reason")}
        for base, got in samples.items():
            if len(got) < ROUND_1:
                continue
            n = sum(got.values())
            group = "4/4" if n == ROUND_1 else "0/4" if n == 0 else "1-3/4"
            examples[base] = {"key": base, **info[base], "group": group, "n_correct_4b": n, "run": Path(run).name}
    return list(examples.values())


def select(examples: list[dict], total: int, wrong_cap: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by = {g: {l: [e for e in examples if e["group"] == g and e["gold"] == l] for l in range(3)}
          for g in ("1-3/4", "0/4", "4/4")}
    chosen = [e for l in range(3) for e in by["1-3/4"][l]]
    target = total / 3
    need = [max(0, round(target - len(by["1-3/4"][l]))) for l in range(3)]
    slots = total - len(chosen)
    scale = slots / sum(need) if sum(need) else 0
    need = [round(n * scale) for n in need]
    need[need.index(max(need))] += slots - sum(need)  # rounding
    wrong_share = min(1.0, wrong_cap / slots) if slots else 0
    for l in range(3):
        n_wrong = min(round(need[l] * wrong_share), len(by["0/4"][l]))
        n_right = min(need[l] - n_wrong, len(by["4/4"][l]))
        chosen += rng.sample(by["0/4"][l], n_wrong) + rng.sample(by["4/4"][l], n_right)
    rng.shuffle(chosen)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--total", type=int, default=10000)
    parser.add_argument("--wrong-cap", type=int, default=2000, help="max always-wrong (0 of 4) examples")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    examples = load_examples()
    chosen = select(examples, args.total, args.wrong_cap, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"prompts_{args.total}.jsonl", "w") as f:
        for e in chosen:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    summary = {
        "available": dict(Counter(e["group"] for e in examples)),
        "selected": len(chosen),
        "by_group": dict(Counter(e["group"] for e in chosen)),
        "by_label": {LABELS[l]: c for l, c in sorted(Counter(e["gold"] for e in chosen).items())},
        "by_group_and_label": {f"{g} {LABELS[l]}": c for (g, l), c in sorted(Counter((e["group"], e["gold"]) for e in chosen).items())},
        "settings": vars(args),
    }
    (OUT_DIR / f"summary_{args.total}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
