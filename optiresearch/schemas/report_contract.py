"""Report contract schema for Phase 68."""
from __future__ import annotations

from optiresearch.memory.schemas import StrictModel


class ReportContract(StrictModel):
    """Contract specifying required structure for a report type."""

    report_contract_id: str
    report_type: str = ""
    exporter_cli: str = ""
    required_sections: list[str] = []
    optional_sections: list[str] = []
    required_tables: list[str] = []
    required_fields: list[str] = []
    linked_artifacts: list[str] = []
    linked_claims: list[str] = []
    safe_wording_required: bool = False
    blocked_claims_section_required: bool = False
    evidence_level_section_required: bool = False
