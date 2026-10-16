# Operations: failures, cleanup and what is not verified

Practical notes from running these jobs: capacity, network, promoted models, deleting resources, and what is still unverified.

## Operational notes

- **Capacity:** two launches in a row had the trainer stuck `PENDING` (36 and 10 minutes) with no reason given; the
  stall watchdog cleaned up each time. Later launches got a GPU within minutes.
- **Network:** run training from a stable connection with the machine awake (`caffeinate`). Connection errors made the
  sampler retry, and one eval stall (81 of 500 requests in 45 minutes) was on a connection that looked like a phone hotspot.
- **Promoted models:** evaluate them with `--template client` ([evaluation.md](evaluation.md)); the server template disables thinking.
- **Deleting:** `scripts/promote_checkpoint.py` first, then delete the trainer; promoted models survive deleting the
  trainer and deployment (verified for both promoted checkpoints).
- **Unverified or not done:** the cookbook's exact advantage call path in the synchronous loop was inferred from a single
  call site; the judge's serverless price was not checked; no teacher-disagreement ablation (keeping those prompts); the
  v4 prompts were not tried on the promoted model; r1 and r2 test rounds were not run. The full pools and teacher traces
  stay out of git (`outputs/` is gitignored); compact pools, run configs and logs, and the test predictions are in
  `reproducibility/` (see its README).
