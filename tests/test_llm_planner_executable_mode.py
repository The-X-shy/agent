"""Tests for executable LLM planning mode."""

import pytest

from optiresearch.agents.llm_planner import LLMPlanner
from optiresearch.agents.prompts.llm_planner_prompt import build_planner_prompt


def test_executable_mode_adds_prompt_instruction():
    context = {
        "objective": "test",
        "allowed_backends": ["phase_to_fft_proxy"],
        "prefer_executable_actions": True,
        "execution_mode": "local",
        "max_candidate_plans": 3,
    }
    messages = build_planner_prompt(context)
    user_content = messages[1]["content"]
    assert "EXECUTABLE ACTIONS REQUIRED" in user_content
    assert "retry_with_smaller_lr" in user_content
    assert "stop_and_report" in user_content


def test_executable_mode_off_omits_instruction():
    context = {
        "objective": "test",
        "allowed_backends": ["phase_to_fft_proxy"],
        "prefer_executable_actions": False,
        "execution_mode": "local",
        "max_candidate_plans": 3,
    }
    messages = build_planner_prompt(context)
    user_content = messages[1]["content"]
    assert "EXECUTABLE ACTIONS REQUIRED" not in user_content


def test_plan_passes_prefer_executable_to_context():
    planner = LLMPlanner()
    context = planner.build_context(
        objective="test",
        allowed_backends=["phase_to_fft_proxy"],
        allowed_task_types=["stable_lens_hsi_codesign"],
        recent_results=[],
        prefer_executable_actions=True,
    )
    assert context["prefer_executable_actions"] is True


def test_plan_default_prefer_executable_is_false():
    planner = LLMPlanner()
    context = planner.build_context(
        objective="test",
        allowed_backends=["phase_to_fft_proxy"],
        allowed_task_types=["stable_lens_hsi_codesign"],
        recent_results=[],
    )
    assert context["prefer_executable_actions"] is False


def test_executable_mode_mock_planner_still_works():
    """Verify that prefer_executable_actions doesn't break mock provider."""
    planner = LLMPlanner()
    result = planner.plan(
        objective="test executable mode",
        provider_name="mock",
        prefer_executable_actions=True,
    )
    assert result.status == "succeeded"
    assert result.selected_proposal is not None
    # Mock proposals are always executable (retry_with_smaller_lr, etc.)
    assert result.selected_proposal.recommended_action != "stop_and_report"


def test_rejected_proposals_are_recorded():
    """Verify rejected proposals appear in validation_errors."""
    planner = LLMPlanner()
    result = planner.plan(
        objective="test",
        provider_name="mock",
    )
    assert result.status == "succeeded"
    assert isinstance(result.rejected_proposals, list)
    assert isinstance(result.validation_errors, list)
