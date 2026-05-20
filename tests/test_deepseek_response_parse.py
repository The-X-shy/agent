import json

from optiresearch.llm.deepseek_provider import DeepSeekProvider


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(
            {
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "message": {
                            "content": "parsed content",
                            "reasoning_content": "internal reasoning",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            }
        ).encode("utf-8")


def test_deepseek_response_parse(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-key")
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())

    response = DeepSeekProvider().complete([{"role": "user", "content": "Hello"}])

    assert response.content == "parsed content"
    assert response.provider == "deepseek"
    assert response.model == "deepseek-v4-pro"
    assert response.finish_reason == "stop"
    assert response.usage == {"prompt_tokens": 2, "completion_tokens": 3}
    assert response.raw["choices"][0]["message"]["reasoning_content"] == "internal reasoning"
