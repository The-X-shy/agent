"""Optional Anthropic provider placeholder."""

from __future__ import annotations

import os
from typing import Any

from optiresearch.llm.base import LLMProvider, LLMProviderError, LLMResponse


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(self) -> None:
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.base_url = os.getenv("ANTHROPIC_BASE_URL")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except Exception:
            return False
        return True

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        raise LLMProviderError("ANTHROPIC_PROVIDER_NOT_BOUND", "Anthropic provider is declared but not bound in MVP tests.")
