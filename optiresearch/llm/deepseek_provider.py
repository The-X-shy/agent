"""DeepSeek provider using the official OpenAI-style chat completions API."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from optiresearch.llm.base import LLMProvider, LLMProviderError, LLMResponse


class DeepSeekProvider(LLMProvider):
    provider_name = "deepseek"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        thinking_type: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-pro"
        self.thinking_type = thinking_type or os.getenv("DEEPSEEK_THINKING_TYPE") or "enabled"
        self.reasoning_effort = reasoning_effort or os.getenv("DEEPSEEK_REASONING_EFFORT") or "high"
        self.timeout = float(timeout if timeout is not None else os.getenv("DEEPSEEK_TIMEOUT", "120"))
        self.max_tokens = os.getenv("DEEPSEEK_MAX_TOKENS")
        self.temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2"))

    def available(self) -> bool:
        return bool(self.api_key)

    def config_summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "available": self.available(),
            "model": self.model,
            "base_url": self.base_url,
            "thinking_type": self.thinking_type,
            "reasoning_effort": self.reasoning_effort,
            "error_code": None if self.available() else "DEEPSEEK_API_KEY_MISSING",
        }

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        if not self.api_key:
            raise LLMProviderError("DEEPSEEK_API_KEY_MISSING", "DEEPSEEK_API_KEY is not configured.")
        model = kwargs.get("model") or self.model
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "thinking": {"type": kwargs.get("thinking_type") or self.thinking_type},
            "reasoning_effort": kwargs.get("reasoning_effort") or self.reasoning_effort,
            "stream": False,
        }
        max_tokens = kwargs.get("max_tokens") or self.max_tokens
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            body["temperature"] = float(temperature)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise LLMProviderError("DEEPSEEK_HTTP_ERROR", f"DeepSeek HTTP error: {exc.code}", {"status": exc.code}) from exc
        except TimeoutError as exc:
            raise LLMProviderError("DEEPSEEK_TIMEOUT", "DeepSeek request timed out.") from exc
        except Exception as exc:
            raise LLMProviderError("DEEPSEEK_HTTP_ERROR", str(exc)) from exc
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        try:
            choice = payload["choices"][0]
            message = choice.get("message", {})
            content = message["content"]
        except Exception as exc:
            raise LLMProviderError("DEEPSEEK_RESPONSE_PARSE_ERROR", "Could not parse DeepSeek response.", {"response": payload}) from exc
        return LLMResponse(
            content=content,
            provider=self.provider_name,
            model=payload.get("model") or model,
            finish_reason=choice.get("finish_reason"),
            usage=payload.get("usage", {}),
            latency_ms=latency_ms,
            raw=payload,
        )
