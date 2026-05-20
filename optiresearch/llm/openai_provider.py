"""Optional OpenAI provider."""

from __future__ import annotations

import os
from typing import Any

from optiresearch.llm.base import LLMProvider, LLMProviderError, LLMResponse


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import openai  # noqa: F401
        except Exception:
            return False
        return True

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        raise LLMProviderError("OPENAI_PROVIDER_NOT_CONFIGURED", "OpenAI SDK binding is optional and not configured for tests.")
