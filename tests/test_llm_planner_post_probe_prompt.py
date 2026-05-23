"""Phase 32: LLM planner post-probe prompt tests."""

from optiresearch.agents.prompts.llm_planner_prompt import (
    build_planner_prompt, SYSTEM_PROMPT, build_mock_proposals,
)


def test_system_prompt_contains_rule_18():
    assert "post_probe_continuation_required" in SYSTEM_PROMPT


def test_system_prompt_contains_run_validated_backend():
    assert "run_validated_backend_experiment" in SYSTEM_PROMPT


def test_continuation_section_added_when_context_present():
    messages = build_planner_prompt({
        "objective": "test",
        "allowed_backends": ["deeplens_geolens_geometric"],
        "recent_results": [
            {"post_probe_continuation_required": True,
             "validated_backend_id": "deeplens_geolens_geometric",
             "validated_backend_evidence_level": "native_lens_simulation"},
        ],
        "prefer_executable_actions": True,
    })
    user_content = messages[1]["content"]
    assert "Post-Probe Continuation Required" in user_content


def test_continuation_not_added_without_context():
    messages = build_planner_prompt({
        "objective": "test",
        "allowed_backends": ["phase_to_fft_proxy"],
        "recent_results": [],
    })
    user_content = messages[1]["content"]
    assert "Post-Probe Continuation Required" not in user_content


def test_mock_proposals_include_continuation():
    actions = [p["recommended_action"] for p in build_mock_proposals()]
    assert "run_validated_backend_experiment" in actions
