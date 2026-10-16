# Experiments and results

The test-set result, the log of every run, what was learned, and the prompt experiments. How each run was built is in [training.md](training.md). Numbers are from the logs under `anli-rl-logs/` and `outputs/` (compact copies are in
[`reproducibility/`](../reproducibility/README.md)); where something is an estimate or was not verified, it says so.

## Evaluation: test_r3 (v3 prompt, thinking on, greedy, 4096 tokens, 1,200 examples)

Promoted model: `scripts/eval_promoted.py --promoted-model accounts/<acct>/models/anli-grpo-v2-step10
--deployment-id <id> --prompt-version zero-shot-v3 --splits test --rounds r3 --temperature 0.0` (client template).
Base model: `anli-eval --backend batch --model accounts/fireworks/models/qwen3-4b --splits test --rounds r3
--temperature 0.0 --max-tokens 4096 --batch-size 1200` (server template, all 1,200 replies had reasoning).

| | Base `qwen3-4b` | `anli-grpo-v2-step10` |
|---|---|---|
| Accuracy | 0.6367 | **0.7008** |
| Macro-F1 | 0.6415 | **0.7012** |
| Recall: entailment / neutral / contradiction | 0.664 / 0.677 / 0.568 | 0.674 / 0.629 / **0.801** |
| Share of predictions: E / N / C | 30 / 42 / 27 | 28 / 33 / 39 |
| Unparsed / errors | 11 / 0 | 1 / 1 |

Paired on the same examples: 167 only the promoted model got right, 90 only the base got right, net +77 (+6.4 points),
McNemar z = 4.7, paired bootstrap 95% interval +3.8 to +9.0 points. The in-training dev numbers for step 10 (0.692 /
0.693) are close to the test result, so the checkpoint choice does not look overfit to dev. Test round r1/r2 were not evaluated.

Gotchas for this eval: requests sent right after a deployment reports READY can fail with 404 "model not found" (8 of
1,200 did; re-running the same command retries only failed rows; the results folder is named after the deployment id, so
copy it when using a new deployment id). A deployment can also end FAILED with no reason in the API; one did, and a retry
on a fresh deployment id worked.


## Experiment log

| Run (log dir) | Setup | Outcome |
|---|---|---|
| `union-1epoch-lr3e-6-hotload900` | 1,493-row pool, 8 x 64, lr 3e-6, 16 steps | Training reward 0.58 -> 0.59; filter ratio rose from about 0.2 to 0.55 (the "groups go uniform" effect); about 45 min, about $20. |
| `n50000-pilot3` | `pool_mixed`, 16 x 48, lr 3e-6 for steps 1-10, then resumed from the step-10 checkpoint at 6e-6 | Dev (500 rows, greedy) flat: 0.658, 0.658, 0.652 / 0.668 (step 10), 0.668 (step 15); `ppo_kl` 0.001 -> 0.0017. Stopped at step 17 when the Mac's network dropped; resources deleted. |
| `n50000-clean3` | `pool_clean` (class-balanced, teacher-verified), 16 x 48, lr 6e-6 | Neutral collapsed in about 5 steps (neutral share of training answers 35% -> 3%, neutral reward 0.61 -> 0.08). Dev 0.654 -> 0.683 accuracy, macro-F1 0.657 -> 0.673, recall 0.73/0.70/0.53 -> 0.88/0.46/0.71. Stopped at step 5 by the filter-ratio guard (0.34, 0.66, 0.69 at steps 3-5). Promoted as `anli-grpo-clean-step5`. |
| `n50000-v2` | `pool_v2`, judge penalty, std floor, class guards, lr 4e-6, 12 steps | Dev 0.656 -> 0.694; test_r3 0.7008 vs base 0.6367 (above). Promoted step 10: `anli-grpo-v2-step10`. |

Failed or abandoned launches (capacity, network) are not listed; see "Operational notes".

## What we learned

1. **Plain GRPO on this pool barely moves dev accuracy.** At 3e-6 and 6e-6 `ppo_kl` stayed near 0.001-0.0017 and dev
   stayed flat. Training reward rose (0.58 -> 0.65) because sampled answers at temperature 1.0 became more consistent
   (the share of all-correct groups roughly tripled, all-wrong stayed near 4%); greedy dev accuracy, which already picks
   the mode, did not change.
2. **Gains are mostly class-prior movement at the neutral boundary.** Row-level comparison of the same 1,000 dev rows:
   - Balanced-pool run, step 0 -> 5: net +29 rows; neutral boundary about +21, entailment<->contradiction flips 0
     (11 fixed, 11 broken), format about +7.
   - v2 run, step 0 -> 12: net +38 rows; neutral boundary +28 (+105 fixed, -77 broken), entailment<->contradiction
     flips +6 (+14 / -8, within noise), format about +4.
   The base model over-answers neutral (42% of predictions vs 34% gold); moving that rate toward the gold share is worth
   roughly +2 to +3 points. Going past it costs neutral recall.
3. **Class mix is a strong lever, and a balanced pool is not neutral.** Entailment and contradiction errors mostly land
   on "neutral", so a 33/33/33 pool pushed neutral down hard, while the original 25/42/33 mix pushed it up. A neutral
   share of about 36% with a lower rate (4e-6) and checkpoints every 2 steps kept all classes at 0.59 or better.
4. **The consistency penalty worked on its target.** Judge-flagged inconsistency (reasoning ends on label A, answer is
   B) fell from about 3% to under 1.5% in 3-4 steps, and invalid answers fell from 0.6% to about 0.1-0.2%. In the
   flagged cases the reasoning was often right (in the 4-case test sample, 3 ended on the gold label: "so the answer is
   neutral", output `contradiction`). The invalid-answer drop is on dev.
5. **Label noise caps what this pool can teach.** The 397B teacher agrees with gold on only about 74% of the mixed pool;
   for 22 of the 57 prompts the 4B got wrong in all 16 samples (39%), every teacher vote also rejected gold. Cleaning the pool by teacher agreement
   changed starting difficulty little (predicted start reward 0.63 vs 0.62). It also removes the prompts the teacher
   rejects, which supplied about 24% of the usable (nonzero-gradient) groups early in the run.
6. **What the promoted model learned (examples, test_r3).** Real reading gains: noticing "She" refers to Val
   (hypothesis "both male" -> contradiction), and separating "played a lady's maid" from "worked as one". Heuristic gains
   with thin reasoning: "important to California's independence" -> contradiction "because it's not mentioned".
   Regressions on gold-neutral items where it now calls an unsupported specific claim ("twelve times", "only practiced in
   3 countries") a contradiction. Overall it adopted a "specific claim the premise doesn't support -> contradiction" habit,
   which explains the contradiction gain (+23 points) and the neutral loss (-5).
7. **Cost control needs automation.** Runs on a laptop stalled on network drops while the GPUs kept billing; the stall
   watchdog and delete-on-failure exist because of that.

## Prompt experiments (base model, dev_r3, 1,000 rows seed 0, thinking on, greedy)

Question: can a prompt fix the base model's weak contradiction recall (0.538, 120 of 325 contradictions answered
"neutral")? Two variants added to `prompts.py`: `zero-shot-v4a` (wider contradiction definition, stricter neutral, "do
not choose neutral because the premise does not mention the claim") and `zero-shot-v4b` (ordered questions:
entailment, then contradiction, then neutral as the remainder). They were tuned on dev, not test.

| Prompt | Accuracy | Macro-F1 | Recall (E / N / C) | Neutral share | Unparsed |
|---|---|---|---|---|---|
| v3 (baseline) | 0.655 | 0.658 | 0.715 / 0.707 / 0.538 | 43% | 7 |
| v4a | 0.633 | 0.638 | 0.653 / 0.743 / 0.498 | 47% | 15 |
| v4b | 0.627 | 0.625 | 0.662 / 0.769 / 0.443 | 51% | 2 |

Both are worse and moved the wrong way: contradiction recall fell and neutral predictions rose (net -22 and -28 rows of
1,000 vs v3; McNemar z 1.8 and 2.1). My guess, not verified: defining neutral as "compatible but not established" gave the
model an easier reading of neutral than the stricter contradiction wording did. Only two variants, one run each. The RL
result (contradiction recall 0.568 -> 0.801 on test) is far larger than anything these prompts did.
