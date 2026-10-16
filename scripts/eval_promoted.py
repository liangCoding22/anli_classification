"""Evaluate a promoted checkpoint through an on-demand Fireworks deployment.

Creates the deployment, waits for READY, runs `anli-eval --template client` against the
deployment's own inference name, then deletes the deployment (unless --keep-deployment).

    PYTHONPATH=src .venv-sft/bin/python scripts/eval_promoted.py \
        --promoted-model accounts/<account>/models/<promoted-id> \
        --deployment-id <promoted-id>-eval \
        --splits dev --rounds r3 --limit 1000

Requires FIREWORKS_API_KEY in the environment (source .env first).
Sampling defaults are --temperature 0.6 --top-p 0.95; pass --temperature 0.0 for greedy decoding. Results go under the
promoted model's folder (not the deployment's); results made with other sampling settings than the folder's cached ones are
refused unless you pass --run-tag NAME (see run_spec.py).
"""

import argparse
import os
import subprocess
import sys

from fireworks.training.sdk.deployment import DeploymentConfig, DeploymentManager

from anli_classification.deploy import serving_probe, wait_until_serving

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHAPE = "accounts/fireworks/deploymentShapes/rft-qwen3-4b"



def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy a promoted model and run anli-eval on it.")
    parser.add_argument("--promoted-model", required=True, help="accounts/<acct>/models/<promoted-id>")
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--prompt-version", default="zero-shot-v3", choices=("zero-shot-v2", "zero-shot-v3"))
    parser.add_argument("--splits", nargs="+", default=["dev"])
    parser.add_argument("--rounds", nargs="+", default=["r3"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", default="outputs", help="root for predictions and metrics")
    parser.add_argument("--max-tokens", type=int, default=4096, help="generation limit per reply")
    parser.add_argument("--keep-deployment", action="store_true", help="do not delete the deployment afterwards")
    parser.add_argument("--run-tag", default=None, help="write results to a separate folder (passed to anli-eval)")
    parser.add_argument("--serve-timeout", type=float, default=600.0,
                        help="seconds to wait for the ready deployment to answer a test request")
    args = parser.parse_args()

    mgr = DeploymentManager(api_key=os.environ["FIREWORKS_API_KEY"])
    cfg = DeploymentConfig(
        deployment_id=args.deployment_id,
        base_model=args.promoted_model,
        deployment_shape=SHAPE,
        expected_deployment_shape=SHAPE,
        enable_hot_load=False,
        hot_load_bucket_type=None,
        for_training=False,
    )

    info = mgr.create_or_get(cfg)
    print("created:", info.name, info.state, flush=True)
    try:
        ready = mgr.wait_for_ready(args.deployment_id)
        # The SDK's inference_model is just the deployment path, which does not address the
        # promoted model. Fireworks wants "<model>#<deployment>" (as in the cookbook).
        model = f"{args.promoted_model}#{ready.inference_model}"
        print("ready:", ready.name, "serving as", model, flush=True)
        # READY does not mean requests are routed yet: early ones can fail with 404 "model not found".
        wait_until_serving(serving_probe(model), timeout_s=args.serve_timeout)
        print("deployment answers requests", flush=True)
        cmd = [
            sys.executable, "-m", "anli_classification.cli",
            "--model", model,
            "--prompt-version", args.prompt_version,
            "--splits", *args.splits,
            "--rounds", *args.rounds,
            "--template", "client",
            "--tokenizer-model", args.tokenizer_model,
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--output-dir", args.output_dir,
            "--max-tokens", str(args.max_tokens),
        ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.run_tag:
            cmd += ["--run-tag", args.run_tag]
        env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, env=env, check=True)
    finally:
        if args.keep_deployment:
            print(f"kept deployment {args.deployment_id}", flush=True)
        else:
            print(f"deleting deployment {args.deployment_id}", flush=True)
            mgr.delete(args.deployment_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
