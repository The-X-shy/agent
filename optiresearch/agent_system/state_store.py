"""Agent state store for Phase 36 — unified state persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from optiresearch.agent_system.events import AgentEvent
from optiresearch.memory.schemas import StrictModel


class AgentState(StrictModel):
    active_objective: str = ""
    current_backend: str = ""
    current_claim_target: str = ""
    last_experiment_result: dict[str, Any] = {}
    last_failure_mode: str = ""
    last_strategy: dict[str, Any] = {}
    known_supported_claims: list[str] = []
    known_unsupported_claims: list[str] = []
    pending_actions: list[str] = []
    available_skills: list[str] = []
    backend_status: dict[str, str] = {}
    remote_worker_status: dict[str, str] = {}
    memory_summary: dict[str, int] = {}
    evidence_summary: dict[str, int] = {}
    snapshot_count: int = 0
    updated_at: float = 0.0


class StateStore:
    def __init__(self, workspace_root: str | Path = "workspace"):
        self._root = Path(workspace_root) / "agent_state"
        self._root.mkdir(parents=True, exist_ok=True)
        self._snapshot_dir = self._root / "snapshots"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._state = AgentState()
        self._snapshots: list[AgentState] = []
        self._load()

    @property
    def state(self) -> AgentState:
        return self._state

    def load(self) -> AgentState:
        self._load()
        return self._state

    def save(self) -> Path:
        self._state.updated_at = time.time()
        self._state.snapshot_count = len(self._snapshots)
        path = self._root / "current_state.json"
        path.write_text(
            json.dumps(self._state.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def update_from_event(self, event: AgentEvent) -> None:
        payload = event.payload or {}
        if event.event_type == "experiment_completed":
            self._state.last_experiment_result = payload
        elif event.event_type == "experiment_failed":
            self._state.last_experiment_result = payload
            if payload.get("error_code"):
                self._state.last_failure_mode = payload.get("failure_mode", payload["error_code"])
        elif event.event_type == "strategy_recommended":
            self._state.last_strategy = payload
        elif event.event_type == "negative_result_recorded":
            fm = payload.get("failure_mode", "")
            if fm:
                self._state.last_failure_mode = fm
            act = payload.get("pending_action", "")
            if act and act not in self._state.pending_actions:
                self._state.pending_actions.append(act)
        elif event.event_type == "claim_checked":
            claim = payload.get("claim_text", "")
            verdict = payload.get("verdict", "")
            if "support" in verdict.lower() and claim not in self._state.known_supported_claims:
                self._state.known_supported_claims.append(claim)
            elif "reject" in verdict.lower() and claim not in self._state.known_unsupported_claims:
                self._state.known_unsupported_claims.append(claim)
        self.save()

    def snapshot(self) -> AgentState:
        snap = self._state.model_copy(deep=True)
        self._snapshots.append(snap)
        snap_path = self._snapshot_dir / f"snapshot_{len(self._snapshots):04d}.json"
        snap_path.write_text(
            json.dumps(snap.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return snap

    def diff_snapshots(self, older: AgentState, newer: AgentState) -> dict[str, Any]:
        diffs: dict[str, Any] = {}
        old_d = older.model_dump(mode="json")
        new_d = newer.model_dump(mode="json")
        for key in set(old_d) | set(new_d):
            if old_d.get(key) != new_d.get(key):
                diffs[key] = {"from": old_d.get(key), "to": new_d.get(key)}
        return diffs

    def export_state_report(self, output_path: str | Path | None = None) -> Path:
        path = Path(output_path or self._root / "state_report.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self._state
        lines = [
            "# Agent State Report",
            "",
            f"**Active Objective:** {s.active_objective}",
            f"**Current Backend:** {s.current_backend}",
            f"**Current Claim Target:** {s.current_claim_target}",
            f"**Last Failure Mode:** {s.last_failure_mode}",
            f"**Snapshot Count:** {s.snapshot_count}",
            f"**Updated:** {s.updated_at}",
            "",
            "## Supported Claims",
            *([f"- {c}" for c in s.known_supported_claims] or ["(none)"]),
            "",
            "## Unsupported Claims",
            *([f"- {c}" for c in s.known_unsupported_claims] or ["(none)"]),
            "",
            "## Pending Actions",
            *([f"- {a}" for a in s.pending_actions] or ["(none)"]),
            "",
            "## Backend Status",
        ]
        for k, v in s.backend_status.items():
            lines.append(f"- {k}: {v}")
        lines.extend(["", "## Remote Workers"])
        for k, v in s.remote_worker_status.items():
            lines.append(f"- {k}: {v}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _load(self) -> None:
        path = self._root / "current_state.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._state = AgentState(**data)
            except Exception:
                pass

    def seed_phase35_result(self) -> None:
        self._state.active_objective = "improve native optical HSI co-design"
        self._state.current_backend = "deeplens_geolens_geometric"
        self._state.last_failure_mode = "unstable_native_geolens_update"
        self._state.pending_actions = ["generate_alternative_plans"]
        self._state.known_supported_claims = [
            "DeepLens native GeoLens geometric HSI co-design path exists (WSL)",
        ]
        self._state.known_unsupported_claims = [
            "Native GeoLens optical parameter improvement",
            "Full coherent wave-optics native HSI co-design",
            "Real HSI performance validation",
        ]
        self.save()

    def update_from_events(self, events: list[AgentEvent]) -> None:
        for event in events:
            self.update_from_event(event)

    def get_latest_snapshot(self) -> AgentState | None:
        return self._snapshots[-1] if self._snapshots else None

    def restore_snapshot(self, snapshot_id: int) -> bool:
        if 0 <= snapshot_id < len(self._snapshots):
            self._state = self._snapshots[snapshot_id].model_copy(deep=True)
            self.save()
            return True
        return False
