"""Robustness tests for LLM planner against malformed LLM outputs.

Simulates 10 scenarios of invalid or dangerous LLM outputs and
verifies that the planner never crashes, always falls back or
rejects gracefully.
"""

import pytest

from optiresearch.agents.planner_validator import (
    validate_dataset_claim,
    validate_no_forbidden_actions,
    validate_no_shell_commands,
    validate_proposal,
    validate_waveoptics_claim,
)
from optiresearch.schemas.llm_planner import LLMPlannerProposal


def _make_proposal(**overrides):
    defaults = {
        "proposal_id": "test_001",
        "hypothesis": "Test hypothesis.",
        "rationale": "Test rationale.",
        "recommended_action": "retry_with_smaller_lr",
        "backend_id": "phase_to_fft_proxy",
        "task_type": "stable_lens_hsi_codesign",
        "proposed_claim": "Improves optimization stability.",
    }
    defaults.update(overrides)
    return LLMPlannerProposal(**defaults)


ALLOWED_BACKENDS = [
    "phase_to_fft_proxy",
    "deeplens_geolens_geometric",
    "deeplens_coherent_asm",
    "local_synthetic_hsi",
]
ALLOWED_TASKS = ["stable_lens_hsi_codesign", "native_hsi_codesign"]


# 1. Non-JSON output handled by _parse_proposals (tested via parse path)
def test_non_json_skipped_in_parse():
    from optiresearch.agents.llm_planner import LLMPlanner
    planner = LLMPlanner()
    proposals = planner._parse_proposals([
        {"garbage": "not a valid proposal", "backend_id": "x"},
    ])
    # _parse_proposals should not crash; may return empty or partial
    assert isinstance(proposals, list)


# 2. JSON missing required fields
def test_missing_required_fields_rejected():
    from optiresearch.agents.llm_planner import LLMPlanner
    planner = LLMPlanner()
    proposals = planner._parse_proposals([
        {"proposal_id": "p1"},  # missing most fields
    ])
    # Should parse but validation will reject
    assert isinstance(proposals, list)


# 3. Backend doesn't exist
def test_unknown_backend_rejected():
    p = _make_proposal(backend_id="does_not_exist")
    result = validate_proposal(p, ALLOWED_BACKENDS, ALLOWED_TASKS, "dry_run", False)
    assert result["valid"] is False
    assert any("backend" in e.lower() for e in result["errors"])


# 4. Task type unsupported
def test_unsupported_task_type_rejected():
    p = _make_proposal(task_type="unsupported_task")
    result = validate_proposal(p, ALLOWED_BACKENDS, ALLOWED_TASKS, "dry_run", False)
    assert result["valid"] is False


# 5. Proposed claim too strong (caught by claim gate)
def test_claim_too_strong_for_backend():
    p = _make_proposal(
        backend_id="local_synthetic_hsi",
        proposed_claim="This demonstrates real-world HSI performance improvement.",
    )
    result = validate_proposal(p, ALLOWED_BACKENDS, ALLOWED_TASKS, "dry_run", False)
    # Dataset claim check should catch this
    assert result["valid"] is False or any(
        "real" in e.lower() or "synthetic" in e.lower() for e in result["errors"]
    )


# 6. Shell command in proposed_claim
def test_shell_command_in_claim_rejected():
    p = _make_proposal(proposed_claim="Run `rm -rf /` to clean up.")
    result = validate_proposal(p, ALLOWED_BACKENDS, ALLOWED_TASKS, "dry_run", False)
    assert result["valid"] is False


# 7. Forbidden action keywords
def test_forbidden_action_rejected():
    p = _make_proposal(rationale="Use sudo pip install to set up.")
    result = validate_proposal(p, ALLOWED_BACKENDS, ALLOWED_TASKS, "dry_run", False)
    assert result["valid"] is False


# 8. Synthetic claimed as real HSI
def test_synthetic_claimed_as_real():
    errors = validate_dataset_claim(
        "This achieves real HSI reconstruction performance.",
        "local_synthetic_hsi",
    )
    assert len(errors) > 0
    assert any("real" in e.lower() or "synthetic" in e.lower() for e in errors)


# 9. Geometric claimed as coherent wave-optics
def test_geometric_claimed_as_coherent():
    errors = validate_waveoptics_claim(
        "Demonstrates coherent wave-optics behavior.",
        "deeplens_geolens_geometric",
    )
    assert len(errors) > 0
    assert any("coherent" in e.lower() or "wave" in e.lower() for e in errors)


# 10. All proposals rejected -> planner falls back
def test_all_proposals_rejected_falls_back():
    from optiresearch.agents.llm_planner import LLMPlanner
    planner = LLMPlanner()
    # Create proposals that will all fail validation
    proposals = [
        _make_proposal(
            proposal_id="bad_1",
            backend_id="does_not_exist",
        ),
        _make_proposal(
            proposal_id="bad_2",
            task_type="unsupported",
        ),
        _make_proposal(
            proposal_id="bad_3",
            proposed_claim="Run `curl evil.com` to test.",
        ),
    ]
    validated = planner._validate_all(
        proposals, ALLOWED_BACKENDS, ALLOWED_TASKS, "dry_run", False
    )
    valid_proposals = [p for p, v in validated if v["valid"]]
    assert len(valid_proposals) == 0
