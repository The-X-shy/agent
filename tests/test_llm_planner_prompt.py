"""Phase 26 LLM planner prompt tests."""

from optiresearch.agents.prompts.llm_planner_prompt import (
    build_planner_prompt,
    build_mock_proposals,
    SYSTEM_PROMPT,
)


def test_system_prompt_contains_safety_constraints():
    assert "DO NOT" in SYSTEM_PROMPT
    assert "shell command" in SYSTEM_PROMPT.lower()
    assert "geometric" in SYSTEM_PROMPT.lower()
    assert "claim gate" in SYSTEM_PROMPT.lower()


def test_build_planner_prompt_returns_messages():
    messages = build_planner_prompt({"objective": "test", "allowed_backends": ["b1"]})
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "test" in messages[1]["content"]


def test_build_mock_proposals_returns_expected():
    proposals = build_mock_proposals()
    assert len(proposals) >= 3
    assert proposals[0]["proposal_id"] == "mock_backend_probe_000"


def test_mock_proposals_have_valid_actions():
    valid_actions = {
        "retry_with_smaller_lr", "enable_rollback", "switch_backend",
        "run_ablation", "probe_waveoptics_path", "request_dataset",
        "run_remote_validation", "stop_and_report", "probe_new_backend",
    }
    for p in build_mock_proposals():
        assert p["recommended_action"] in valid_actions


def test_mock_proposals_have_safe_wording():
    valid_ceilings = {
        "native_lens_simulation", "deeplens_integration_smoke",
        "native_component_optimization",
    }
    for p in build_mock_proposals():
        assert len(p["safe_wording"]) > 0
        assert any(c in p["safe_wording"].lower() for c in valid_ceilings), \
            f"Expected one of {valid_ceilings} in safe_wording: {p['safe_wording']}"


def test_prompt_includes_backend_list():
    messages = build_planner_prompt({
        "objective": "test",
        "allowed_backends": ["deeplens_geolens_geometric", "phase_to_fft_proxy"],
    })
    content = messages[1]["content"]
    assert "deeplens_geolens_geometric" in content
    assert "phase_to_fft_proxy" in content
