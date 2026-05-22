"""Test that LLM planner prompt includes metric trajectory constraints."""

import pytest
from optiresearch.agents.prompts.llm_planner_prompt import (
    build_planner_prompt,
    _build_metric_trajectory_summary,
)


def test_prompt_includes_metric_trajectory_constraints():
    context = {
        "objective": "test",
        "allowed_backends": ["phase_to_fft_proxy"],
        "execution_mode": "local",
        "prefer_executable_actions": True,
    }
    messages = build_planner_prompt(context)
    system = messages[0]["content"]
    assert "claim downgrade" in system.lower() or "claim_downgraded" in system.lower()
    assert "measurable metrics" in system.lower()


def test_prompt_includes_backend_ceiling_info():
    context = {
        "objective": "test",
        "allowed_backends": ["phase_to_fft_proxy"],
        "execution_mode": "local",
    }
    messages = build_planner_prompt(context)
    system = messages[0]["content"]
    assert "native_full_reconstruction_proxy" in system


def test_prompt_discourages_early_stop():
    context = {
        "objective": "test",
        "allowed_backends": ["phase_to_fft_proxy"],
        "execution_mode": "local",
        "prefer_executable_actions": True,
    }
    messages = build_planner_prompt(context)
    system = messages[0]["content"]
    assert "do not stop" in system.lower() or "do NOT stop" in system


def test_metric_trajectory_summary_with_results():
    results = [
        {
            "status": "succeeded",
            "result_payload": {
                "reconstruction_loss_before": 0.10,
                "reconstruction_loss_after": 0.05,
                "improvement_detected": True,
            },
        },
        {
            "status": "succeeded",
            "result_payload": {
                "reconstruction_loss_before": 0.05,
                "reconstruction_loss_after": 0.04,
                "improvement_detected": True,
            },
        },
    ]
    summary = _build_metric_trajectory_summary(results)
    assert "0.100000" in summary
    assert "0.050000" in summary
    assert "True" in summary
    assert "Iter" in summary


def test_metric_trajectory_summary_empty():
    assert _build_metric_trajectory_summary([]) == ""


def test_prompt_includes_metric_trajectory_when_results_present():
    context = {
        "objective": "test",
        "allowed_backends": ["phase_to_fft_proxy"],
        "execution_mode": "local",
        "recent_results": [
            {
                "status": "succeeded",
                "result_payload": {
                    "reconstruction_loss_after": 0.05,
                    "improvement_detected": True,
                },
            },
        ],
    }
    messages = build_planner_prompt(context)
    user = messages[1]["content"]
    assert "Metric Trajectory" in user
    assert "0.050000" in user


def test_prompt_returns_two_messages():
    context = {"objective": "test"}
    messages = build_planner_prompt(context)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
