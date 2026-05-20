from datetime import datetime, timezone

import pytest

from optiresearch.memory.meta_trace import MetaTraceWriter
from optiresearch.memory.schemas import MetaTrace, make_trace_id
from optiresearch.storage.sqlite_store import SQLiteStore


def _trace(task: str = "plan first mock run") -> MetaTrace:
    trace_id = make_trace_id("ws", "run-1", "step-1", "LeadInvestigator", task)
    return MetaTrace(
        trace_id=trace_id,
        workspace_id="ws",
        run_id="run-1",
        branch_id=None,
        step_id="step-1",
        actor="LeadInvestigator",
        phase="Explore",
        task=task,
        skill_id=None,
        skill_version=None,
        tool=None,
        input_refs=[],
        output_refs=[],
        findings=["decision: use mock DeepLens for first pass"],
        limitations=[],
        next_action="build optical spec",
        status="succeeded",
        timestamp_start=datetime.now(timezone.utc),
        timestamp_end=datetime.now(timezone.utc),
        parents=[],
        content_hash=None,
        metadata={"objective": "mock EDOF-HSI"},
    )


def test_write_read_and_append_only_trace(tmp_path):
    store = SQLiteStore(tmp_path / "memory.sqlite")
    store.init_db()
    writer = MetaTraceWriter(store)

    written = writer.write_trace(_trace())
    loaded = writer.get_trace(written.trace_id)
    repeated = writer.write_trace(_trace())

    assert loaded == written
    assert repeated == written
    assert len(writer.list_traces(run_id="run-1")) == 1


def test_append_only_trace_rejects_conflicting_payload(tmp_path):
    store = SQLiteStore(tmp_path / "memory.sqlite")
    store.init_db()
    writer = MetaTraceWriter(store)
    original = writer.write_trace(_trace())
    conflicting = _trace(task="plan first mock run")
    conflicting.findings = ["decision: changed after write"]

    with pytest.raises(ValueError, match="append-only"):
        writer.write_trace(conflicting)

    assert writer.get_trace(original.trace_id) == original
