"""Build a label-cleaned, class-balanced RL pool from the mixed pool, using the teacher's votes.

Why: the 397B teacher disagrees with gold on about 21% of the mixed pool (all of its votes against gold), and on
about 40% of the prompts the 4B fails every time. Those prompts are mostly label noise or real ambiguity, so
they teach nothing. The mixed pool also has far fewer entailment rows than neutral or contradiction, and
training drifted away from entailment.

Keeps a prompt when every parsed teacher vote equals gold ("agree"). Drops "disagree" and "split". The
balanced file then keeps an equal number of prompts per gold label (random, seeded).

    PYTHONPATH=src python3 scripts/build_clean_pool.py

Output (next to the input pool): pool_clean.jsonl (balanced), pool_clean_unbalanced.jsonl, clean_summary.json.
Each row is the original pool row plus `teacher_votes`.

With `--neutral-share` (and optionally `--hard`) it also writes pool_v2.jsonl: entailment and contradiction kept at the
balanced count, neutral raised to that share of the pool (the balanced pool let neutral collapse in five steps), and,
with `--hard`, the "trusted hard" prompts added: the 4B got 0 of 4 and every teacher vote agrees with gold.
"""

import argparse
import glob
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("entailment", "neutral", "contradiction")
POOL_DIR = Path("outputs/qwen3-4b/zero-shot-v3/training_pool/train_r3_n50000_seed0_excl")


def load_votes(trace_glob: str) -> dict:
    votes = defaultdict(list)
    for path in glob.glob(trace_glob):
        for line in open(path):
            trace = json.loads(line)
            if trace["pred"] and trace["unparsed_reason"] is None:
                votes[trace["key"]].append(trace["pred"])
    return votes


def teacher_verdict(votes: list, gold_label: str) -> str:
    if not votes:
        return "uncovered"
    agreeing = sum(vote == gold_label for vote in votes)
    return "agree" if agreeing == len(votes) else ("split" if agreeing else "disagree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pool", type=Path, default=POOL_DIR / "pool_mixed.jsonl")
    parser.add_argument("--traces", default="outputs/teacher_traces/*/*/traces.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neutral-share", type=float, default=None, help="also write pool_v2.jsonl with this neutral share")
    parser.add_argument("--hard", action="store_true", help="add 4B-0/4 prompts the teacher agrees with to pool_v2")
    parser.add_argument("--prompts", default="outputs/teacher_prompts/*.jsonl")
    args = parser.parse_args()

    votes = load_votes(args.traces)
    rows = [json.loads(line) for line in args.pool.read_text().splitlines() if line.strip()]
    verdicts = Counter()
    kept = defaultdict(list)
    for row in rows:
        label = LABELS[row["gold"]]
        verdict = teacher_verdict(votes.get(row["key"], []), label)
        verdicts[(label, verdict)] += 1
        if verdict == "agree":
            kept[row["gold"]].append({**row, "teacher_votes": votes[row["key"]]})

    rng = random.Random(args.seed)
    per_class = min(len(v) for v in kept.values())
    balanced = []
    for gold in sorted(kept):
        rows_for_class = list(kept[gold])
        rng.shuffle(rows_for_class)
        balanced += rows_for_class[:per_class]
    rng.shuffle(balanced)
    unbalanced = [row for gold in sorted(kept) for row in kept[gold]]
    rng.shuffle(unbalanced)

    out_dir = args.pool.parent
    v2_summary = None
    if args.neutral_share is not None:
        hard = {}
        if args.hard:
            in_pool = {row["key"] for row in rows}
            for path in glob.glob(args.prompts):
                for line in open(path):
                    r = json.loads(line)
                    if (r.get("n_correct_4b") == 0 and r["key"] not in in_pool and r["key"] not in hard
                            and teacher_verdict(votes.get(r["key"], []), LABELS[r["gold"]]) == "agree"):
                        hard[r["key"]] = {**r, "n_correct": 0, "n_samples": 4, "teacher_votes": votes[r["key"]]}
        by_class = {g: [row for row in balanced if row["gold"] == g] for g in range(3)}
        hard_by_class = defaultdict(list)
        for row in hard.values():
            hard_by_class[row["gold"]].append(row)
        # hard prompts are almost all entailment; top contradiction (the weakest dev class) up to the same count
        entail_total = len(by_class[0]) + len(hard_by_class[0])
        contra_have = len(by_class[2]) + len(hard_by_class[2])
        used = {row["key"] for row in by_class[2]}
        extra_contra = [row for row in kept[2] if row["key"] not in used]
        rng.shuffle(extra_contra)
        by_class[2] = by_class[2] + extra_contra[:max(0, entail_total - contra_have)]
        e_c = sum(len(by_class[g]) + len(hard_by_class[g]) for g in (0, 2))
        n_neutral = round(args.neutral_share / (1 - args.neutral_share) * e_c)
        neutral_pool = list(kept[1])
        rng.shuffle(neutral_pool)
        n_neutral = min(n_neutral, len(neutral_pool))
        v2 = by_class[0] + by_class[2] + neutral_pool[:n_neutral] + list(hard.values())
        rng.shuffle(v2)
        (out_dir / "pool_v2.jsonl").write_text("".join(json.dumps(row) + "\n" for row in v2))
        v2_summary = {"rows": len(v2), "by_label": {LABELS[g]: sum(r["gold"] == g for r in v2) for g in range(3)},
                      "neutral_share": round(sum(r["gold"] == 1 for r in v2) / len(v2), 3), "hard_rows": len(hard),
                      "hard_by_label": {LABELS[g]: len(hard_by_class[g]) for g in range(3)}}
    for name, data in (("pool_clean.jsonl", balanced), ("pool_clean_unbalanced.jsonl", unbalanced)):
        (out_dir / name).write_text("".join(json.dumps(row) + "\n" for row in data))

    def passes(data):
        return {LABELS[g]: dict(sorted(Counter(r["n_correct"] for r in data if r["gold"] == g).items())) for g in range(3)}

    summary = {
        "input_rows": len(rows),
        "teacher_verdicts": {f"{label}/{verdict}": n for (label, verdict), n in sorted(verdicts.items())},
        "kept_by_label": {LABELS[g]: len(v) for g, v in sorted(kept.items())},
        "balanced_per_class": per_class,
        "balanced_rows": len(balanced),
        "unbalanced_rows": len(unbalanced),
        "balanced_n_correct_of_4_by_label": passes(balanced),
        "seed": args.seed,
        "traces": args.traces,
        "pool_v2": v2_summary,
    }
    (out_dir / "clean_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
