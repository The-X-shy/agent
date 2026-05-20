"""LLM provider registry."""

from __future__ import annotations

import os

from optiresearch.llm.anthropic_provider import AnthropicProvider
from optiresearch.llm.deepseek_provider import DeepSeekProvider
from optiresearch.llm.local_provider import LocalProvider
from optiresearch.llm.mock_provider import MockLLMProvider
from optiresearch.llm.openai_provider import OpenAIProvider


def get_llm_provider(name: str | None = None):
    if name:
        return _provider_by_name(name)
    env_name = os.getenv("OPTIRESEARCH_LLM_PROVIDER")
    if env_name:
        return _provider_by_name(env_name)
    for provider in (DeepSeekProvider(), OpenAIProvider(), AnthropicProvider(), LocalProvider()):
        if provider.available():
            return provider
    return MockLLMProvider()


def list_llm_providers() -> list[dict]:
    providers = [DeepSeekProvider(), OpenAIProvider(), AnthropicProvider(), LocalProvider(), MockLLMProvider()]
    return [
        {
            "provider": provider.provider_name,
            "available": provider.available(),
            "model": getattr(provider, "model", None),
            "base_url": getattr(provider, "base_url", getattr(provider, "url", None)),
            "thinking_type": getattr(provider, "thinking_type", None),
            "reasoning_effort": getattr(provider, "reasoning_effort", None),
        }
        for provider in providers
    ]


def _provider_by_name(name: str):
    normalized = name.lower().replace("-", "_")
    if normalized == "deepseek":
        return DeepSeekProvider()
    if normalized == "openai":
        return OpenAIProvider()
    if normalized == "anthropic":
        return AnthropicProvider()
    if normalized == "local":
        return LocalProvider()
    if normalized == "mock":
        return MockLLMProvider()
    raise ValueError(f"Unknown LLM provider: {name}")
