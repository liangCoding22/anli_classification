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
from pathlib import Path

from anli_classification.paths import default_pool_path, teacher_prompts_glob, teacher_traces_glob
from anli_classification.pools import build_clean_pools, find_hard_prompts, load_votes, summarize


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pool", type=Path, default=default_pool_path(),
                        help="the mixed pool; outputs are written next to it (default: under $ANLI_OUTPUT_DIR or outputs/)")
    parser.add_argument("--traces", default=teacher_traces_glob())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neutral-share", type=float, default=None, help="also write pool_v2.jsonl with this neutral share")
    parser.add_argument("--hard", action="store_true", help="add 4B-0/4 prompts the teacher agrees with to pool_v2")
    parser.add_argument("--prompts", default=teacher_prompts_glob(),
                        help="prompt files to find hard prompts in (read in sorted order, so the result does not depend on the "
                             "directory listing)")
    parser.add_argument("--hard-candidates", type=Path, default=None,
                        help="ordered hard prompts to use instead of searching --prompts; "
                             "reproducibility/pools/hard_candidates.compact.jsonl rebuilds the pool_v2 of the last run exactly")
    args = parser.parse_args()
    if not args.pool.exists():
        parser.error(f"{args.pool} does not exist; pass --pool (or set ANLI_OUTPUT_DIR)")

    votes = load_votes(args.traces)
    rows = [json.loads(line) for line in args.pool.read_text().splitlines() if line.strip()]
    hard = {}
    if args.neutral_share is not None and args.hard_candidates is not None:
        candidates = [json.loads(line) for line in args.hard_candidates.read_text().splitlines() if line.strip()]
        hard = {row["key"]: row for row in candidates}
    elif args.neutral_share is not None and args.hard:
        prompt_rows = (json.loads(line) for path in sorted(glob.glob(args.prompts)) for line in open(path))
        hard = find_hard_prompts(prompt_rows, {row["key"] for row in rows}, votes)
    pools = build_clean_pools(rows, votes, args.seed, args.neutral_share, hard)

    out_dir = args.pool.parent
    if pools.v2 is not None:
        write_jsonl(out_dir / "pool_v2.jsonl", pools.v2)
    write_jsonl(out_dir / "pool_clean.jsonl", pools.balanced)
    write_jsonl(out_dir / "pool_clean_unbalanced.jsonl", pools.unbalanced)
    summary = summarize(pools, len(rows), args.seed, args.traces)
    (out_dir / "clean_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
