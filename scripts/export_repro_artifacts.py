"""Export the small, shareable files needed to reproduce the reported results into `reproducibility/`.

The big inputs (pools with sampled reasoning, teacher traces) stay out of git; this keeps what is needed to rebuild
or check them: compact pools (prompt text, gold, derived stats), every teacher vote, run configs and metrics, compact
rollouts, compact eval predictions, and a manifest with the sha256 of each large local file.

    PYTHONPATH=src python3 scripts/export_repro_artifacts.py \
        --pool-dir outputs/qwen3-4b/zero-shot-v3/training_pool/train_r3_n50000_seed0_excl \
        --traces-glob 'outputs/teacher_traces/*/*/traces.jsonl' \
        --runs anli-rl-logs/n50000-v2 anli-rl-logs/n50000-clean3 \
        --evals promoted=outputs/anli-grpo-v2-step10-test/<run>/zero-shot-v3-client-template \
                base=outputs/anli-base-test-batch/qwen3-4b/zero-shot-v3 \
        --out reproducibility

Pool text is ANLI (CC BY-NC 4.0, https://huggingface.co/datasets/facebook/anli); see reproducibility/README.md.
"""

import argparse
import csv
import glob
import gzip
import hashlib
import json
import platform
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

POOL_FILES = ("pool_mixed.jsonl", "pool_clean.jsonl", "pool_v2.jsonl")
POOL_FIELDS = ("key", "gold", "premise", "hypothesis", "reason", "group", "n_correct", "n_samples", "teacher_votes")
RUN_SMALL_FILES = (
    "run_config.json", "run_config.run1-lr3e-6.json", "metrics.jsonl", "metrics.run1-lr3e-6.jsonl",
    "stdout.log", "stdout.run1-lr3e-6.log", "trainer_job.json", "dataloader.json", "summary.txt",
)
ROLLOUT_COLUMNS = ("key", "split", "gold", "pred", "reward", "label_reward", "judge_verdict", "well_formed", "thinking_chars")
PACKAGES = ("fireworks-ai", "openai", "torch", "transformers", "datasets", "tokenizers")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
            n += 1
    return n


def teacher_votes(traces_glob: str) -> dict:
    votes = {}
    for name in sorted(glob.glob(traces_glob)):
        for trace in read_jsonl(Path(name)):
            entry = votes.setdefault(trace["key"], {"key": trace["key"], "gold": trace["gold"], "votes": []})
            if trace.get("pred") and trace.get("unparsed_reason") is None:
                entry["votes"].append(trace["pred"])
    return votes


def export_pools(pool_dir: Path, votes: dict, out: Path, manifest: dict) -> None:
    for name in POOL_FILES:
        src = pool_dir / name
        if not src.exists():
            continue
        rows = ({**{f: r.get(f) for f in POOL_FIELDS if f != "teacher_votes"},
                 "teacher_votes": r.get("teacher_votes") or votes.get(r["key"], {}).get("votes")}
                for r in read_jsonl(src))
        dst = out / "pools" / name.replace(".jsonl", ".compact.jsonl")
        n = write_jsonl(dst, rows)
        manifest["pools"][name] = {"rows": n, "sha256_full_local_file": sha256(src), "compact_file": str(dst.relative_to(out))}
    for name in ("clean_summary.json", "summary.json"):
        if (pool_dir / name).exists():
            shutil.copy(pool_dir / name, out / "pools" / ("n50000_" + name if name == "summary.json" else name))
    n = write_jsonl(out / "pools" / "teacher_votes.jsonl", votes.values())
    manifest["teacher_votes"] = {"prompts": n}


def export_run(run_dir: Path, out: Path, manifest: dict) -> None:
    dst = out / "runs" / run_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    kept = []
    for name in RUN_SMALL_FILES:
        if (run_dir / name).exists():
            shutil.copy(run_dir / name, dst / name)
            kept.append(name)
    for rollouts in sorted(run_dir.glob("rollouts*.jsonl")):
        target = dst / (rollouts.stem + ".compact.csv.gz")
        with gzip.open(target, "wt", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(ROLLOUT_COLUMNS)
            for r in read_jsonl(rollouts):
                writer.writerow([r.get(c, "") if r.get(c) is not None else "" for c in ROLLOUT_COLUMNS])
        kept.append(target.name)
        manifest["runs"].setdefault(run_dir.name, {})[rollouts.name + "_sha256_full_local_file"] = sha256(rollouts)
    manifest["runs"].setdefault(run_dir.name, {})["files"] = kept


def export_eval(tag: str, eval_dir: Path, out: Path, manifest: dict) -> None:
    dst = out / "results" / tag
    dst.mkdir(parents=True, exist_ok=True)
    if (eval_dir / "metrics.json").exists():
        shutil.copy(eval_dir / "metrics.json", dst / "metrics.json")
    for pred_file in sorted(eval_dir.glob("*_r[123].jsonl")):
        latest = {}  # the store is append-only: a retried key appears twice and the last line wins
        for r in read_jsonl(pred_file):
            latest[r["key"]] = {"key": r["key"], "gold": r["gold"], "pred": r.get("pred"), "error": bool(r.get("error"))}
        rows = latest.values()
        n = write_jsonl(dst / (pred_file.stem + ".compact.jsonl"), rows)
        manifest["results"][tag] = {"slice": pred_file.stem, "rows": n, "source_dir": str(eval_dir)}


def command_output(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def package_versions() -> dict:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--traces-glob", required=True)
    parser.add_argument("--extra-hash", nargs="*", default=[], help="more large local files to record a sha256 for")
    parser.add_argument("--runs", nargs="*", default=[], type=Path)
    parser.add_argument("--evals", nargs="*", default=[], help="tag=directory with <slice>.jsonl and metrics.json")
    parser.add_argument("--cookbook-dir", type=Path, default=None, help="cookbook checkout, to record its commit")
    parser.add_argument("--extra-json", type=Path, default=None, help="extra environment facts to record")
    parser.add_argument("--out", type=Path, default=Path("reproducibility"))
    args = parser.parse_args()

    manifest = {"pools": {}, "runs": {}, "results": {}, "large_inputs": {}}
    votes = teacher_votes(args.traces_glob)
    export_pools(args.pool_dir, votes, args.out, manifest)
    for name in sorted(glob.glob(args.traces_glob)) + list(args.extra_hash):
        manifest["large_inputs"][name] = sha256(Path(name))
    for run in args.runs:
        export_run(run, args.out, manifest)
    for item in args.evals:
        tag, _, directory = item.partition("=")
        export_eval(tag, Path(directory), args.out, manifest)
    manifest["environment"] = {
        "python": platform.python_version(),
        "packages": package_versions(),
        "repo_commit": command_output("git", "rev-parse", "HEAD"),
        "cookbook_commit": command_output("git", "-C", str(args.cookbook_dir), "rev-parse", "HEAD") if args.cookbook_dir else "",
    }
    if args.extra_json:
        manifest["environment"].update(json.loads(args.extra_json.read_text()))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"pools": {k: v["rows"] for k, v in manifest["pools"].items()}, "runs": list(manifest["runs"]),
                      "results": list(manifest["results"]), "large_inputs": len(manifest["large_inputs"])}, indent=2))


if __name__ == "__main__":
    main()
