"""LLM call audit helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from optiresearch.llm.base import LLMResponse


SECRET_ENV_KEYS = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY")


def redact_secrets(text: str) -> str:
    redacted = text
    for key in SECRET_ENV_KEYS:
        value = os.getenv(key)
        if value:
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def build_llm_trace_metadata(
    prompt: str,
    response: LLMResponse | None,
    structured_schema: str | None,
    fallback_used: bool,
    error_code: str | None = None,
) -> dict[str, Any]:
    response_content = response.content if response else ""
    return {
        "llm_used": response is not None and not fallback_used,
        "llm_provider": response.provider if response else None,
        "llm_model": response.model if response else None,
        "prompt_hash": _sha256(redact_secrets(prompt)),
        "response_hash": _sha256(redact_secrets(response_content)),
        "structured_schema": structured_schema,
        "fallback_used": fallback_used,
        "error_code": error_code,
        "latency_ms": response.latency_ms if response else None,
        "usage": response.usage if response else {},
    }


def record_llm_call(prompt: str, response: LLMResponse, output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    artifact = path / f"llm_call_{_sha256(prompt)[:12]}.json"
    artifact.write_text(
        json.dumps(
            {
                "prompt": redact_secrets(prompt),
                "response": redact_secrets(response.content),
                "provider": response.provider,
                "model": response.model,
                "usage": response.usage,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
