"""Toy benchmark cases for future OptiMemoryBench work."""

from __future__ import annotations


def load_toy_tasks() -> list[dict[str, str]]:
    return [
        {
            "task_id": "mock-edof-hsi",
            "objective": "Design a mock EDOF-HSI optical encoder",
            "expected_intent": "deeplens psf evidence",
        }
    ]
