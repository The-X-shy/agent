"""Runtime state containers for the MVP flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeState:
    """Small mutable state object shared by rule-based agents."""

    workspace_id: str
    run_id: str
    objective: str
    trace_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
