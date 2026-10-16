"""Teacher-verified training pools.

The mixed pool holds prompts the 4B answered right 1-3 times out of 4. A 397B teacher answered each prompt (2 votes). A prompt
is kept when every vote agrees with gold; the kept prompts are balanced across classes, and `pool_v2` can also tilt the pool
towards neutral and add "trusted hard" prompts (the 4B got 0 of 4 and the teacher agrees with gold).

Pure functions: no file or network access, so the selection can be tested and its random order reproduced exactly (the
order of every `rng` call below is part of what a given seed produces).
"""

from __future__ import annotations

import glob
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

LABELS = ("entailment", "neutral", "contradiction")


def load_votes(trace_glob: str) -> dict:
    """Every parsed teacher vote per prompt key, from the files matching `trace_glob` (read in sorted order)."""
    votes = defaultdict(list)
    for path in sorted(glob.glob(trace_glob)):
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


def find_hard_prompts(prompt_rows: Iterable[dict], in_pool_keys: set, votes: dict) -> dict:
    """Prompts the 4B got 0 of 4 on, not already in the pool, that every teacher vote agrees with (first occurrence wins)."""
    hard: dict = {}
    for r in prompt_rows:
        if (r.get("n_correct_4b") == 0 and r["key"] not in in_pool_keys and r["key"] not in hard
                and teacher_verdict(votes.get(r["key"], []), LABELS[r["gold"]]) == "agree"):
            hard[r["key"]] = {**r, "n_correct": 0, "n_samples": 4, "teacher_votes": votes[r["key"]]}
    return hard


@dataclass
class CleanPools:
    verdicts: Counter
    kept: dict
    per_class: int
    balanced: list
    unbalanced: list
    v2: Optional[list] = None
    v2_summary: Optional[dict] = None


def build_clean_pools(
    rows: list,
    votes: dict,
    seed: int = 0,
    neutral_share: Optional[float] = None,
    hard: Optional[dict] = None,
) -> CleanPools:
    """Filter by teacher agreement, balance classes, and (with `neutral_share`) build pool_v2. Deterministic for a seed."""
    verdicts = Counter()
    kept = defaultdict(list)
    for row in rows:
        label = LABELS[row["gold"]]
        verdict = teacher_verdict(votes.get(row["key"], []), label)
        verdicts[(label, verdict)] += 1
        if verdict == "agree":
            kept[row["gold"]].append({**row, "teacher_votes": votes[row["key"]]})

    rng = random.Random(seed)
    per_class = min(len(v) for v in kept.values())
    balanced = []
    for gold in sorted(kept):
        rows_for_class = list(kept[gold])
        rng.shuffle(rows_for_class)
        balanced += rows_for_class[:per_class]
    rng.shuffle(balanced)
    unbalanced = [row for gold in sorted(kept) for row in kept[gold]]
    rng.shuffle(unbalanced)

    result = CleanPools(verdicts, kept, per_class, balanced, unbalanced)
    if neutral_share is not None:
        hard = hard or {}
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
        n_neutral = round(neutral_share / (1 - neutral_share) * e_c)
        neutral_pool = list(kept[1])
        rng.shuffle(neutral_pool)
        n_neutral = min(n_neutral, len(neutral_pool))
        v2 = by_class[0] + by_class[2] + neutral_pool[:n_neutral] + list(hard.values())
        rng.shuffle(v2)
        result.v2 = v2
        result.v2_summary = {
            "rows": len(v2), "by_label": {LABELS[g]: sum(r["gold"] == g for r in v2) for g in range(3)},
            "neutral_share": round(sum(r["gold"] == 1 for r in v2) / len(v2), 3), "hard_rows": len(hard),
            "hard_by_label": {LABELS[g]: len(hard_by_class[g]) for g in range(3)},
        }
    return result


def summarize(pools: CleanPools, input_rows: int, seed: int, traces: str) -> dict:
    def passes(data):
        return {LABELS[g]: dict(sorted(Counter(r["n_correct"] for r in data if r["gold"] == g).items())) for g in range(3)}

    return {
        "input_rows": input_rows,
        "teacher_verdicts": {f"{label}/{verdict}": n for (label, verdict), n in sorted(pools.verdicts.items())},
        "kept_by_label": {LABELS[g]: len(v) for g, v in sorted(pools.kept.items())},
        "balanced_per_class": pools.per_class,
        "balanced_rows": len(pools.balanced),
        "unbalanced_rows": len(pools.unbalanced),
        "balanced_n_correct_of_4_by_label": passes(pools.balanced),
        "seed": seed,
        "traces": traces,
        "pool_v2": pools.v2_summary,
    }
