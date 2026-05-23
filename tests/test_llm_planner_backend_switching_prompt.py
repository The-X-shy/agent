"""Test LLM planner prompt includes backend switching instructions."""

import pytest
from optiresearch.agents.prompts.llm_planner_prompt import build_planner_prompt


def test_prompt_includes_backend_switching_guidance():
    context = {
        "objective": "test",
        "allowed_backends": ["phase_to_fft_proxy", "deeplens_geolens_geometric"],
        "execution_mode": "local",
        "prefer_executable_actions": True,
    }
    messages = build_planner_prompt(context)
    system = messages[0]["content"]
    assert "switch" in system.lower()
    assert "backend" in system.lower()


def test_prompt_mentions_claim_ceiling_reached():
    context = {
        "objective": "test",
        "allowed_backends": ["phase_to_fft_proxy"],
        "execution_mode": "local",
    }
    messages = build_planner_prompt(context)
    system = messages[0]["content"]
    assert "claim_ceiling_reached" in system or "claim ceiling" in system.lower()


def test_prompt_discourages_same_backend_repetition():
    context = {
        "objective": "test",
        "allowed_backends": ["phase_to_fft_proxy", "deeplens_geolens_geometric"],
        "execution_mode": "local",
        "prefer_executable_actions": True,
    }
    messages = build_planner_prompt(context)
    user = messages[1]["content"]
    system = messages[0]["content"]
    combined = user + system
    assert "higher" in combined.lower() or "progression" in combined.lower() or "switch" in combined.lower()


def test_backend_progression_section_with_multiple_backends():
    context = {
        "objective": "test",
        "allowed_backends": ["phase_to_fft_proxy", "deeplens_geolens_geometric"],
        "execution_mode": "local",
    }
    messages = build_planner_prompt(context)
    user = messages[1]["content"]
    assert "Backend Progression Available" in user
    assert "phase_to_fft_proxy" in user
    assert "deeplens_geolens_geometric" in user


def test_no_backend_progression_with_single_backend():
    context = {
        "objective": "test",
        "allowed_backends": ["phase_to_fft_proxy"],
        "execution_mode": "local",
    }
    messages = build_planner_prompt(context)
    user = messages[1]["content"]
    assert "Backend Progression Available" not in user
