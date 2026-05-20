"""Local HTTP provider placeholder."""

from __future__ import annotations

import os
from typing import Any

from optiresearch.llm.base import LLMProvider, LLMProviderError, LLMResponse


class LocalProvider(LLMProvider):
    provider_name = "local"

    def __init__(self) -> None:
        self.url = os.getenv("OPTIRESEARCH_LOCAL_LLM_URL")
        self.model = os.getenv("OPTIRESEARCH_LOCAL_LLM_MODEL", "local")

    def available(self) -> bool:
        return bool(self.url)

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        raise LLMProviderError("LOCAL_PROVIDER_NOT_BOUND", "Local LLM HTTP binding is not configured.")
