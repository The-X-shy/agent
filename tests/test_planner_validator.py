"""Phase 26 planner validator tests."""

from optiresearch.schemas.llm_planner import LLMPlannerProposal
from optiresearch.agents.planner_validator import (
    validate_proposal,
    validate_no_shell_commands,
    validate_no_forbidden_actions,
    validate_dataset_claim,
    validate_waveoptics_claim,
)


def _make_proposal(**kwargs) -> LLMPlannerProposal:
    defaults = {
        "proposal_id": "test_1",
        "recommended_action": "retry_with_smaller_lr",
        "backend_id": "deeplens_geolens_geometric",
        "task_type": "stable_lens_hsi_codesign",
        "risk_level": "low",
    }
    defaults.update(kwargs)
    return LLMPlannerProposal(**defaults)


def test_valid_proposal_passes():
    p = _make_proposal()
    result = validate_proposal(p, ["deeplens_geolens_geometric"], ["stable_lens_hsi_codesign"], "dry_run")
    assert result["valid"] is True
    assert len(result["errors"]) == 0


def test_unknown_backend_fails():
    p = _make_proposal(backend_id="nonexistent")
    result = validate_proposal(p, ["deeplens_geolens_geometric"], ["stable_lens_hsi_codesign"], "dry_run")
    assert result["valid"] is False


def test_unknown_task_type_fails():
    p = _make_proposal(task_type="nonexistent")
    result = validate_proposal(p, ["deeplens_geolens_geometric"], ["stable_lens_hsi_codesign"], "dry_run")
    assert result["valid"] is False


def test_shell_commands_rejected():
    result = validate_no_shell_commands("run `ls` or $(echo test)")
    assert len(result) > 0


def test_forbidden_actions_rejected():
    result = validate_no_forbidden_actions("we should run git or rm the file or sudo make")
    assert len(result) > 0


def test_forbidden_actions_rejected_curl():
    result = validate_no_forbidden_actions("use curl to download data")
    assert len(result) > 0


def test_synthetic_backend_real_claim_rejected():
    result = validate_dataset_claim("real HSI performance", "local_synthetic_hsi")
    assert len(result) > 0


def test_geometric_backend_coherent_claim_rejected():
    result = validate_waveoptics_claim("full wave-optics native HSI co-design", "deeplens_geolens_geometric")
    assert len(result) > 0


def test_clean_text_passes():
    assert validate_no_shell_commands("Reduce optical learning rate") == []
    assert validate_no_forbidden_actions("Run stable training with rollback") == []


def test_empty_claim_passes():
    assert validate_dataset_claim("", "deeplens_geolens_geometric") == []
    assert validate_waveoptics_claim("", "phase_to_fft_proxy") == []


def test_valid_backends_pass_dataset_check():
    assert validate_dataset_claim("real performance metrics", "deeplens_geolens_geometric") == []
