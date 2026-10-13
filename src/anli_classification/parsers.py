from __future__ import annotations

import re
from typing import Optional

from anli_classification.schemas import LABEL2ID


class LabelParser:
    """Maps a model reply to a label id by finding the first NLI label word in it."""

    _PATTERN = re.compile(r"\b(entailment|neutral|contradiction)\b", re.IGNORECASE)

    def parse(self, text: Optional[str]) -> Optional[int]:
        match = self._PATTERN.search(text or "")
        return LABEL2ID[match.group(1).lower()] if match else None
