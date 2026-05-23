"""Agent event bus for Phase 36 — pub/sub event coordination."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from optiresearch.agent_system.events import AgentEvent, EventType


EventHandler = Callable[[AgentEvent], None]


class EventBus:
    def __init__(self):
        self._events: list[AgentEvent] = []
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)

    def publish(self, event: AgentEvent) -> None:
        self._events.append(event)
        for handler in self._subscribers.get(event.event_type, []):
            try:
                handler(event)
            except Exception:
                pass
        for handler in self._subscribers.get("*", []):
            try:
                handler(event)
            except Exception:
                pass

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    def list_events(self) -> list[AgentEvent]:
        return list(self._events)

    def filter_events(
        self,
        event_type: str | None = None,
        source_module: str | None = None,
        severity: str | None = None,
        related_run_id: str | None = None,
    ) -> list[AgentEvent]:
        results = self._events
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if source_module:
            results = [e for e in results if e.source_module == source_module]
        if severity:
            results = [e for e in results if e.severity == severity]
        if related_run_id:
            results = [e for e in results if e.related_run_id == related_run_id]
        return results

    def export_events(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.model_dump(mode="json") for e in self._events]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def clear(self) -> None:
        self._events.clear()

    def count(self) -> int:
        return len(self._events)

    def latest(self) -> AgentEvent | None:
        return self._events[-1] if self._events else None


# Global singleton
_default_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus
