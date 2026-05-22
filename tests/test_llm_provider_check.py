"""Tests for LLM provider environment check."""

import os
from unittest.mock import MagicMock, patch

import pytest

from optiresearch.agents.llm_provider_check import check_llm_provider


def test_no_key_returns_skipped(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = check_llm_provider("deepseek")
    assert result["status"] == "skipped"
    assert result["error_code"] == "DEEPSEEK_API_KEY_MISSING"
    assert result["provider"] == "deepseek"


def test_mock_provider_available():
    result = check_llm_provider("mock")
    assert result["status"] == "available"
    assert result["provider"] == "mock"
    assert result["error_code"] is None


def test_with_key_http_success(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")

    mock_response = MagicMock()
    mock_response.read.return_value = (
        b'{"choices":[{"message":{"content":"OK"},"finish_reason":"stop"}],'
        b'"model":"deepseek-v4-pro","usage":{}}'
    )
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        with patch("urllib.request.Request", return_value=MagicMock()):
            result = check_llm_provider("deepseek")
    assert result["status"] == "available", f"Expected available, got {result['status']}: {result.get('error_message')}"
    assert result["provider"] == "deepseek"
    assert result["error_code"] is None


def test_with_key_http_error(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")

    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
        "https://api.deepseek.com", 401, "Unauthorized", {}, None
    )):
        with patch("urllib.request.Request", return_value=MagicMock()):
            result = check_llm_provider("deepseek")
    assert result["status"] == "provider_error"
    assert result["error_code"] == "DEEPSEEK_HTTP_ERROR"
