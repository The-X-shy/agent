"""Base contracts for LLM providers."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field


class LLMProviderError(Exception):
    """Controlled provider failure used by agents for rule fallback."""

    def __init__(self, error_code: str, message: str, raw: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.raw = raw or {}

    def to_dict(self) -> dict[str, Any]:
        return {"error_code": self.error_code, "message": self.message, "raw": self.raw}


class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    finish_reason: Optional[str] = None
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[float] = None
    raw: Optional[dict[str, Any]] = None


class LLMProvider:
    provider_name = "base"
    model = "unknown"

    def available(self) -> bool:
        return False

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        raise LLMProviderError("LLM_PROVIDER_NOT_IMPLEMENTED", "Provider does not implement complete().")

    def structured_complete(self, messages: list[dict[str, str]], schema: type[BaseModel], **kwargs: Any) -> BaseModel:
        constrained = [
            {
                "role": "system",
                "content": "You must respond with valid JSON only. Do not wrap it in markdown. Do not include explanations outside JSON.",
            },
            *messages,
        ]
        response = self.complete(constrained, **kwargs)
        try:
            return schema.model_validate(_parse_json_object(response.content))
        except Exception as first_error:
            repair_messages = [
                {
                    "role": "system",
                    "content": "Repair the following invalid JSON into valid JSON matching this schema. Return JSON only.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "schema": schema.model_json_schema(),
                            "invalid_output": response.content,
                            "validation_error": str(first_error),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            repair = self.complete(repair_messages, **kwargs)
            try:
                return schema.model_validate(_parse_json_object(repair.content))
            except Exception as repair_error:
                raise LLMProviderError(
                    f"{self.provider_name.upper()}_STRUCTURED_OUTPUT_ERROR",
                    "Structured output could not be parsed or repaired.",
                    {"first_error": str(first_error), "repair_error": str(repair_error)},
                ) from repair_error


def _parse_json_object(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
