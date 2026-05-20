from optiresearch.llm.structured_output import ClaimReviewDraft
from optiresearch.memory.claim_evidence import ClaimEvidenceManager
from optiresearch.storage.sqlite_store import SQLiteStore


class OverclaimingProvider:
    provider_name = "overclaim"
    model = "mock"

    def available(self):
        return True

    def structured_complete(self, messages, schema, **kwargs):
        return ClaimReviewDraft(
            claim_text="unsupported claim",
            suggested_status="supported",
            reasoning="LLM wants support",
            required_caveats=[],
            missing_evidence=[],
            follow_up_experiments=[],
            risk_level="low",
        )


def test_llm_claim_review_cannot_override_evidence_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "claims.sqlite"))
    manager = ClaimEvidenceManager(SQLiteStore())
    claim = manager.create_claim("The encoder is physically validated.", scope={"backend": "deeplens"})

    draft = manager.review_claim_with_llm(claim.claim_id, provider=OverclaimingProvider())
    reviewed = manager.review_claim(claim.claim_id)

    assert draft.suggested_status == "supported"
    assert reviewed.status == "unsupported"
