from optiresearch.llm.deepseek_provider import DeepSeekProvider


def test_deepseek_provider_defaults_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    provider = DeepSeekProvider()

    assert provider.available() is False
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.model == "deepseek-v4-pro"
    assert provider.thinking_type == "enabled"
    assert provider.reasoning_effort == "high"


def test_deepseek_provider_available_with_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    provider = DeepSeekProvider()

    assert provider.available() is True
    assert provider.config_summary()["model"] == "deepseek-v4-pro"
    assert provider.config_summary()["base_url"] == "https://api.deepseek.com"
