"""LLM provider environment check.

Validates LLM provider availability and returns a structured
status report without exposing API keys or credentials.
"""

from __future__ import annotations

import time
from typing import Any

from optiresearch.llm.audit import redact_secrets
from optiresearch.llm.base import LLMProviderError
from optiresearch.llm.registry import get_llm_provider


def check_llm_provider(provider_name: str = "deepseek") -> dict[str, Any]:
    """Check LLM provider availability and health.

    Returns a structured result with status, model, and error
    information. API keys and credentials are never included in
    the output.
    """
    try:
        provider = get_llm_provider(provider_name)
    except Exception as exc:
        return {
            "status": "provider_error",
            "provider": provider_name,
            "model": None,
            "base_url": None,
            "error_code": "PROVIDER_NOT_FOUND",
            "error_message": str(exc),
            "latency_ms": None,
        }

    config = provider.config_summary()
    base_url = redact_secrets(str(config.get("base_url", "")))

    if not provider.available():
        return {
            "status": "skipped",
            "provider": config.get("provider", provider_name),
            "model": config.get("model"),
            "base_url": base_url,
            "error_code": config.get("error_code", "PROVIDER_NOT_AVAILABLE"),
            "error_message": "Provider is not available. Check API key configuration.",
            "latency_ms": None,
        }

    # Attempt a minimal API call to verify connectivity
    start = time.perf_counter()
    try:
        response = provider.complete([
            {"role": "user", "content": "Reply with OK."}
        ])
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return {
            "status": "available",
            "provider": response.provider,
            "model": response.model,
            "base_url": base_url,
            "error_code": None,
            "error_message": None,
            "latency_ms": latency_ms,
        }
    except LLMProviderError as exc:
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return {
            "status": "provider_error",
            "provider": config.get("provider", provider_name),
            "model": config.get("model"),
            "base_url": base_url,
            "error_code": exc.error_code,
            "error_message": str(exc),
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return {
            "status": "provider_error",
            "provider": config.get("provider", provider_name),
            "model": config.get("model"),
            "base_url": base_url,
            "error_code": "PROVIDER_CHECK_FAILED",
            "error_message": str(exc),
            "latency_ms": latency_ms,
        }
