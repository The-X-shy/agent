"""Phase 26 planner trace tests."""

from optiresearch.agents.planner_trace import (
    start_planner_trace,
    record_context,
    record_response,
    record_validation,
    record_selection,
    finalize_trace,
    list_planner_traces,
    inspect_planner_trace,
)


def test_start_and_finalize_trace():
    trace = start_planner_trace("test_trace_01")
    record_context(trace, {"objective": "test"})
    record_response(trace, [{"proposal": "mock"}])
    record_validation(trace, [])
    record_selection(trace, {"selected": "yes"})
    path = finalize_trace(trace)
    assert "test_trace_01" in path
    assert "workspace/planner_traces" in path


def test_trace_writes_files():
    trace = start_planner_trace("test_trace_files")
    record_context(trace, {"objective": "test file"})
    path = finalize_trace(trace)
    import os
    assert os.path.exists(path)
    assert os.path.exists(f"{path}/context_summary.json")
    assert os.path.exists(f"{path}/_trace_index.json")


def test_list_traces():
    traces = list_planner_traces()
    assert isinstance(traces, list)


def test_inspect_nonexistent_trace():
    result = inspect_planner_trace("nonexistent_trace_xyz")
    assert result is None
