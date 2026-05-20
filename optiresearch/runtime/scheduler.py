"""Scheduler placeholder for future long-running jobs."""

from __future__ import annotations


class Scheduler:
    """Minimal synchronous scheduler placeholder."""

    def submit(self, name: str) -> dict[str, str]:
        return {"job_id": name, "status": "submitted"}
