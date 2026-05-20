from optiresearch.llm.audit import build_llm_trace_metadata, record_llm_call, redact_secrets
from optiresearch.llm.base import LLMResponse


def test_llm_audit_redacts_keys_and_builds_hash_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value")
    prompt = "key=secret-value"
    response = LLMResponse(content="result", provider="mock", model="mock-model", usage={"total_tokens": 1})

    redacted = redact_secrets(prompt)
    metadata = build_llm_trace_metadata(prompt, response, "ResearchPlanDraft", fallback_used=False)
    artifact = record_llm_call(prompt, response, tmp_path)

    assert "secret-value" not in redacted
    assert metadata["llm_used"] is True
    assert metadata["prompt_hash"]
    assert metadata["response_hash"]
    assert artifact.exists()
    assert "secret-value" not in artifact.read_text(encoding="utf-8")
