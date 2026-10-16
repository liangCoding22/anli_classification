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
import json
from collections import Counter
from pathlib import Path

from anli_classification.paths import output_root, training_pool_root
from anli_classification.schemas import LABELS
from anli_classification.teacher_data import load_examples, select_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--total", type=int, default=10000)
    parser.add_argument("--wrong-cap", type=int, default=2000, help="max always-wrong (0 of 4) examples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-dir", type=Path, default=training_pool_root(),
                        help="directory with one subdirectory per anli-training-pool run")
    parser.add_argument("--out-dir", type=Path, default=output_root() / "teacher_prompts")
    args = parser.parse_args()

    examples = load_examples(str(args.pool_dir))
    chosen = select_prompts(examples, args.total, args.wrong_cap, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / f"prompts_{args.total}.jsonl", "w") as f:
        for e in chosen:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    summary = {
        "available": dict(Counter(e["group"] for e in examples)),
        "selected": len(chosen),
        "by_group": dict(Counter(e["group"] for e in chosen)),
        "by_label": {LABELS[l]: c for l, c in sorted(Counter(e["gold"] for e in chosen).items())},
        "by_group_and_label": {f"{g} {LABELS[l]}": c for (g, l), c in sorted(Counter((e["group"], e["gold"]) for e in chosen).items())},
        "settings": {"total": args.total, "wrong_cap": args.wrong_cap, "seed": args.seed},
    }
    (args.out_dir / f"summary_{args.total}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
