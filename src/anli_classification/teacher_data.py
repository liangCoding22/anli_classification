"""Teacher data: choosing prompts for the teacher, building its requests, collecting its traces, filtering traces for SFT.

Pure helpers moved out of `scripts/` so they can be tested; the scripts keep the file and network handling. The function
bodies are unchanged from the scripts they came from (`select` is now `select_prompts`, and `load_examples` takes the pool
directory as an argument instead of a module constant).
"""

from __future__ import annotations

import glob
import json
import random
import re
from pathlib import Path

from anli_classification.llm.base import ChatRequest
from anli_classification.parsers import parse_answer, split_reasoning, unparsed_reason
from anli_classification.prompts import NliPromptV3
from anli_classification.schemas import LABELS, Example

ROUND_1 = 4  # answers per example in the first sampling round


def load_examples(pool_dir: str) -> list[dict]:
    """One record per example with a complete, error-free round 1, and its number of correct answers."""
    examples: dict[str, dict] = {}
    for run in sorted(glob.glob(f"{pool_dir}/train_r3_n*")):
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


def select_prompts(examples: list[dict], total: int, wrong_cap: int, seed: int) -> list[dict]:
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


STRAY = re.compile(r"\.cw\s*$")
CJK = re.compile(r"[\u4e00-\u9fff]")
# Claims about dataset or annotation conventions ("standard NLI protocol", "many NLI datasets label ...").
PROTOCOL_CLAIM = re.compile(
    r"(NLI (datasets?|benchmarks?|protocol|conventions?)|standard NLI (protocol|practice|convention)|in (most|many|typical) NLI)"
)


def clean(reasoning: str) -> str:
    return STRAY.sub(".", reasoning.rstrip()).strip()


def usable(t: dict, max_tokens: int, max_waits: int, drop_cjk: bool = False, drop_protocol_claims: bool = False) -> bool:
    return (
        "error" not in t
        and t["correct"]
        and t["finish_reason"] == "stop"
        and t["completion_tokens"] <= max_tokens
        and len(re.findall(r"\bWait\b", t["reasoning"])) <= max_waits
        and bool(t["reasoning"].strip())
        and not (drop_cjk and CJK.search(t["reasoning"]))
        and not (drop_protocol_claims and PROTOCOL_CLAIM.search(t["reasoning"]))
    )


def build_requests(rows: list[dict], default_samples: int, max_tokens: int, temperature: float, top_p: float):
    prompt = NliPromptV3()
    requests = []
    for r in rows:
        example = Example(split="train", round_id="r3", uid=r["key"], premise=r["premise"],
                          hypothesis=r["hypothesis"], label=r["gold"], reason="")
        messages = prompt.build(example)
        for s in range(r.get("samples", default_samples)):
            requests.append(ChatRequest(key=f"{r['key']}#{s}", messages=messages, max_tokens=max_tokens,
                                        temperature=temperature, top_p=top_p))
    return requests


def collect(run_dir: Path, gold_of: dict[str, int]) -> int:
    n = 0
    with open(run_dir / "traces.jsonl", "w") as out:
        for output in sorted(run_dir.glob("chunk_*/output.jsonl")):
            for line in output.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                custom_id = row["custom_id"]
                key, _, sample = custom_id.rpartition("#")
                if row.get("error"):
                    rec = {"key": key, "sample": int(sample), "error": json.dumps(row["error"])}
                else:
                    resp = row.get("response") or row
                    choice = resp["choices"][0]
                    message = choice["message"]
                    text = message.get("content") or ""
                    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                    if not reasoning:
                        reasoning, text = split_reasoning(text)
                    pred = parse_answer(text)
                    rec = {"key": key, "sample": int(sample), "gold": gold_of[key], "pred": LABELS[pred] if pred is not None else None,
                           "correct": pred == gold_of[key], "finish_reason": choice.get("finish_reason"),
                           "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"),
                           "unparsed_reason": unparsed_reason(text), "raw": text, "reasoning": reasoning.strip()}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n
