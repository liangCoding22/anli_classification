"""Client-side chat templating, so an eval sends the model the same prompt tokens RL trained on.

A promoted RL checkpoint is not served with the base model's chat template: Fireworks registers
it with `useHfApplyChatTemplate: false`, and its built-in qwen3 template prefills an empty
`<think>\\n\\n</think>\\n\\n`, i.e. non-thinking mode. The cookbook renderer used in training
leaves thinking on. Rendering the prompt here with the HF tokenizer's own template
(`enable_thinking=True`) and sending it to /completions skips the server's template entirely.

Chat backend only (needs a deployment of the model). The batch API accepts only `messages`
(`prompt_token_ids` rows are rejected with HTTP 400), and for a promoted model it ignores
`chat_template_kwargs` and rejects `reasoning_effort` ("non-reasoning model").

Needs `transformers` and `jinja2` (`pip install -e ".[template]"`); imported lazily.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-4B"  # the RL run's --tokenizer-model


class ChatTemplateRenderer:
    """Renders chat messages to a generation prompt with the model's HF chat template."""

    def __init__(self, tokenizer_model: str = DEFAULT_TOKENIZER_MODEL, enable_thinking: bool = True) -> None:
        self.tokenizer_model = tokenizer_model
        self.enable_thinking = enable_thinking
        self._tokenizer: Optional[Any] = None

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise SystemExit(
                    'client-side templating needs transformers and jinja2: pip install -e ".[template]"'
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)
        return self._tokenizer

    def render(self, messages: Sequence[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
