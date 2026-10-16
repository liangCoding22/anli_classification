"""The single-turn rollout the cookbook loop calls: sample, grade, log."""

from __future__ import annotations

from typing import Any
from typing import Optional

from anli_classification.parsers import is_well_formed, LabelParser, unparsed_reason
from anli_classification.rl.defaults import EVAL_TEMPERATURE, MALFORMED_MARGIN, NO_REASONING
from anli_classification.rl.logs import FormatTracker, RolloutLog
from anli_classification.rl.reward import ConsistencyJudge, label_reward, thinking_text, training_reward
from anli_classification.schemas import LABELS


def make_rollout_fn(
    setup: Any,
    *,
    renderer_name: str = "",
    rollout_log: Optional[RolloutLog] = None,
    format_tracker: Optional[FormatTracker] = None,
    eval_formats: Optional[dict[str, str]] = None,
    eval_temperature: Optional[float] = EVAL_TEMPERATURE,
    eval_preds: Optional[dict[str, Optional[int]]] = None,
    consistency_judge: Optional[ConsistencyJudge] = None,
    consistency_penalty: float = 0.0,
    reward_version: str = "v2",
    malformed_margin: float = MALFORMED_MARGIN,
) -> Any:
    """Single-turn rollout: one sampled answer per call, graded by `label_reward`.

    `evaluation` is True for dev evaluation rollouts, which sample at `eval_temperature` (None keeps the
    training temperature); each graded rollout is written to
    `rollout_log` with its completion text, so failures can be inspected later.
    Mirrors the cookbook's DeepMath example; the cookbook imports are local to this function.
    """
    import asyncio

    from training.renderer import get_text_content
    from training.utils import build_renderer, prepare_sampling_messages
    from training.utils.rl.rollout import single_turn_renderer_rollout

    renderer = build_renderer(setup.tokenizer, setup.tokenizer_id, renderer_name, load_image_processor=False)
    sampler = setup.sampler  # borrowed from the recipe; do not close
    eval_sample_kwargs = (
        setup.sample_kwargs if eval_temperature is None else {**setup.sample_kwargs, "temperature": eval_temperature}
    )

    async def rollout_fn(row: dict, evaluation: bool = False) -> Any:
        async def grade(row_: dict, message: Any, _parse_success: bool) -> float:
            completion = get_text_content(message)
            label_value = await asyncio.to_thread(label_reward, completion, row_["gold"])
            pred = LabelParser().parse(completion)
            thinking = thinking_text(message)
            verdict = None
            if consistency_judge is not None and not evaluation and pred is not None and thinking:
                verdict = await asyncio.to_thread(
                    consistency_judge.verdict, row_.get("premise", ""), row_.get("hypothesis", ""), thinking, LABELS[pred]
                )
            if (not evaluation and consistency_judge is not None and reward_version == "v3"
                    and consistency_penalty > 0 and pred is not None and not thinking):
                verdict = NO_REASONING
                consistency_judge.record(NO_REASONING)
            # evaluation never gets a judge or a penalty: it is the plain label reward
            reward = training_reward(label_value, pred, verdict, 0.0 if evaluation else consistency_penalty,
                                     reward_version, malformed_margin)
            well_formed = is_well_formed(completion)
            parse_status = unparsed_reason(completion) or "parsed"
            if format_tracker is not None and not evaluation:
                format_tracker.record(well_formed)
            if eval_formats is not None and evaluation:
                eval_formats[row_["key"]] = parse_status
            if eval_preds is not None and evaluation:
                eval_preds[row_["key"]] = pred
            if rollout_log is not None:
                rollout_log.write({
                    "split": "eval" if evaluation else "train",
                    "key": row_["key"],
                    "gold": row_["gold"],
                    "pred": pred,
                    "reward": reward,
                    "label_reward": label_value,
                    "judge_verdict": verdict,
                    "well_formed": well_formed,
                    "parse_status": parse_status,
                    "outcome": "correct" if label_value == 1.0 else ("incorrect" if well_formed else "invalid"),
                    "completion": completion,
                    "thinking": thinking,
                    "thinking_chars": len(thinking),
                })
            return reward

        return await single_turn_renderer_rollout(
            row,
            renderer=renderer,
            sample_with_prompt_tokens=sampler.sample_with_prompt_tokens,
            message_builder=lambda r: prepare_sampling_messages(r.get("messages", [])),
            reward_fn=grade,
            sample_kwargs=eval_sample_kwargs if evaluation else setup.sample_kwargs,
            tokenizer=setup.tokenizer,
        )

    return rollout_fn
