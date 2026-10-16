"""Build chat-format SFT data from teacher traces (`generate_teacher_traces.py` output).

    PYTHONPATH=src .venv-sft/bin/python scripts/build_distill_sft.py \
        --traces outputs/teacher_traces/qwen3p5-397b-a17b/batch1/traces.jsonl \
        --prompts outputs/teacher_prompts/batch1_prompts.jsonl \
        --output-dir outputs/teacher_sft/batch1

A trace is kept when it
- reaches the gold label with exactly one label word as the answer and finished normally (not cut off),
- is at most --max-tokens long (the student must be able to finish within its inference limit),
- has at most --max-waits "Wait" restarts (heavy back-and-forth teaches dithering),
- and, with the optional flags, has no Chinese characters (--drop-cjk) and makes no claim about NLI
  dataset conventions (--drop-protocol-claims). Filters apply per trace, so a prompt can still be kept
  through another correct trace.

The teacher's thinking often ends with a stray ".cw" after its label (a leaked marker, about half of
the replies); it is stripped. Per prompt, --per-prompt traces are kept (shortest first). Rows have the
same shape as sft_data.py: system, user (zero-shot-v3 prompt), assistant with `reasoning_content`.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from anli_classification.schemas import LABELS
from anli_classification.sft_data import to_messages
from anli_classification.teacher_data import clean, usable

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--prompts", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-waits", type=int, default=8)
    parser.add_argument("--per-prompt", type=int, default=1)
    parser.add_argument("--pick", choices=("shortest", "longest", "random"), default="shortest",
                        help="which usable traces to keep when a prompt has more than --per-prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop-cjk", action="store_true", help="drop traces containing Chinese characters")
    parser.add_argument("--drop-protocol-claims", action="store_true",
                        help='drop traces that claim an NLI dataset/protocol convention ("standard NLI protocol", ...)')
    args = parser.parse_args()

    prompts = {}
    for path in args.prompts:
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                prompts[r["key"]] = r
    by_key: dict[str, list[dict]] = defaultdict(list)
    seen = Counter()
    for path in args.traces:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            seen["answers"] += 1
            if usable(t, args.max_tokens, args.max_waits, args.drop_cjk, args.drop_protocol_claims):
                by_key[t["key"]].append(t)
    seen["usable_answers"] = sum(len(v) for v in by_key.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    kept = []
    with open(args.output_dir / "train_sft.jsonl", "w") as f, open(args.output_dir / "train_sft_meta.jsonl", "w") as meta:
        for key in sorted(by_key):
            p = prompts[key]
            ts = by_key[key]
            if args.pick == "random":
                ts = rng.sample(ts, len(ts))
            else:
                ts = sorted(ts, key=lambda t: t["completion_tokens"], reverse=args.pick == "longest")
            for t in ts[: args.per_prompt]:
                record = {"key": key, "premise": p["premise"], "hypothesis": p["hypothesis"], "gold": p["gold"],
                          "reason": "", "trace": {"reasoning": clean(t["reasoning"])}}
                f.write(json.dumps({"messages": to_messages(record)}, ensure_ascii=False) + "\n")
                meta.write(json.dumps({"key": key, "group": p["group"], "gold": LABELS[p["gold"]],
                                       "tokens": t["completion_tokens"], "sample": t["sample"]}) + "\n")
                kept.append((p["group"], LABELS[p["gold"]], t["completion_tokens"]))

    tokens = sorted(k[2] for k in kept)
    summary = {
        "prompts_in_file": len(prompts),
        "answers": seen["answers"],
        "usable_answers": seen["usable_answers"],
        "prompts_with_usable_trace": len(by_key),
        "rows_written": len(kept),
        "by_group": dict(Counter(k[0] for k in kept)),
        "by_label": dict(Counter(k[1] for k in kept)),
        "tokens": {"mean": sum(tokens) // max(1, len(tokens)), "median": tokens[len(tokens) // 2] if tokens else 0,
                   "p90": tokens[int(0.9 * len(tokens))] if tokens else 0, "max": tokens[-1] if tokens else 0},
        "settings": {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v) for k, v in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
