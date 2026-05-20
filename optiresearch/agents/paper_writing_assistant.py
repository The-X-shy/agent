"""LLM-assisted paper writing assistant with evidence-safe citations."""

from __future__ import annotations

from typing import Any

from optiresearch.llm.registry import get_llm_provider
from optiresearch.llm.structured_output import PaperSectionDraft


class PaperWritingAssistant:
    def draft_experiment_section(self, report_inputs: dict[str, Any], provider=None) -> PaperSectionDraft:
        provider = provider or get_llm_provider()
        artifact_ids = list(report_inputs.get("artifact_ids", []))
        claim_ids = list(report_inputs.get("claim_ids", []))
        if getattr(provider, "available", lambda: False)():
            try:
                draft = provider.structured_complete(
                    [
                        {"role": "system", "content": "Draft a paper section. Cite only supplied IDs. Return JSON only."},
                        {"role": "user", "content": str(report_inputs)},
                    ],
                    PaperSectionDraft,
                    artifact_ids=artifact_ids,
                    claim_ids=claim_ids,
                )
                return draft.model_copy(
                    update={
                        "cited_artifact_ids": [item for item in draft.cited_artifact_ids if item in artifact_ids],
                        "cited_claim_ids": [item for item in draft.cited_claim_ids if item in claim_ids],
                    }
                )
            except Exception:
                pass
        return PaperSectionDraft(
            section_title="Experiment Summary",
            paragraphs=[str(report_inputs.get("summary", "Registered artifacts and reviewed claims are summarized."))],
            cited_artifact_ids=artifact_ids,
            cited_claim_ids=claim_ids,
            limitations=["LLM text is not evidence."],
            unsupported_claim_warnings=[],
        )
