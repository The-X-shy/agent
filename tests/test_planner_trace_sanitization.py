"""Tests for planner trace sanitization."""

import json
import os
from pathlib import Path

import pytest

from optiresearch.agents.planner_trace import (
    PlannerTrace,
    finalize_trace,
    record_context,
    record_response,
    redact_api_keys,
    redact_authorization_headers,
    redact_env_values,
    start_planner_trace,
)


def test_redact_api_keys_replaces_secrets(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-deepseek-123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-openai-456")

    data = {
        "prompt": "Use key sk-secret-deepseek-123 to authenticate.",
        "nested": {"token": "Bearer sk-secret-openai-456"},
        "list": ["prefix sk-secret-deepseek-123 suffix"],
    }
    result = redact_api_keys(data)
    assert "sk-secret-deepseek-123" not in str(result)
    assert "sk-secret-openai-456" not in str(result)
    assert "[REDACTED]" in str(result)


def test_redact_authorization_headers():
    data = {
        "Authorization": "Bearer sk-abc123",
        "x-api-key": "secret-key-xyz",
        "Content-Type": "application/json",
        "nested": {"Authorization": "Bearer sk-nested"},
    }
    result = redact_authorization_headers(data)
    assert result["Authorization"] == "[REDACTED]"
    assert result["x-api-key"] == "[REDACTED]"
    assert result["Content-Type"] == "application/json"
    assert result["nested"]["Authorization"] == "[REDACTED]"


def test_redact_env_values_is_alias(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
    data = {"text": "key is sk-test-123 here"}
    result = redact_env_values(data)
    assert "sk-test-123" not in str(result)


def test_full_trace_no_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-full-789")
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    trace = start_planner_trace("test_sanitize_001")
    record_context(trace, {
        "objective": "test",
        "api_key": "should-be-removed",
        "DEEPSEEK_API_KEY": "sk-secret-full-789",
        "nested": {"token": "sk-secret-full-789"},
    })
    record_response(trace, [
        {"content": "Response with key sk-secret-full-789 embedded."}
    ])
    finalize_trace(trace)

    # Read all files and verify no secrets
    trace_dir = workspace / "planner_traces" / "test_sanitize_001"
    for f in trace_dir.iterdir():
        if f.suffix == ".json":
            content = f.read_text(encoding="utf-8")
            assert "sk-secret-full-789" not in content, f"Secret found in {f.name}"
            assert "[REDACTED]" in content or "should-be-removed" not in content


def test_record_response_sanitized(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-resp-secret")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()

    trace = start_planner_trace("test_resp_sanitize")
    record_response(trace, [
        {
            "choices": [{
                "message": {
                    "content": '{"proposals":[{"proposal_id":"p1","key":"sk-resp-secret"}]}'
                }
            }]
        }
    ])
    path = trace.output_dir / "raw_response.json"
    content = path.read_text(encoding="utf-8")
    assert "sk-resp-secret" not in content
