"""Toy tasks for OptiMemoryBench."""

from __future__ import annotations


def load_tasks() -> list[dict[str, str]]:
    return [
        {
            "task_id": "recipe-reuse",
            "task_type": "DeepLens-Recipe-Reuse",
            "objective": "Design a mock EDOF-HSI optical encoder with reusable plan",
        },
        {
            "task_id": "claim-qa",
            "task_type": "EDOF-HSI-Claim-QA",
            "question": "What evidence supports the depth stability claim?",
        },
        {
            "task_id": "skill-load",
            "task_type": "Skill-Load-Efficiency",
            "intent": "simulate psf and inspect artifact metrics",
        },
    ]
