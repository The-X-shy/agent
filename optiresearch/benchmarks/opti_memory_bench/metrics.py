"""Benchmark metric helpers."""

from __future__ import annotations


def bool_score(value: bool) -> int:
    return 1 if value else 0


def precision(selected: set[str], expected: set[str]) -> float:
    if not selected:
        return 0.0
    return round(len(selected & expected) / len(selected), 6)


def recall(selected: set[str], expected: set[str]) -> float:
    if not expected:
        return 1.0
    return round(len(selected & expected) / len(expected), 6)
