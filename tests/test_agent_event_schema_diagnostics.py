"""Test AgentEvent schema with new diagnostic event types and source modules."""

from optiresearch.agent_system.events import AgentEvent, EventType, SourceModule


class TestDiagnosticEventTypes:
    def test_diagnosis_started(self):
        event = AgentEvent.create("diagnosis_started", "analyzer", {"diagnosis_id": "test"})
        assert event.event_type == "diagnosis_started"
        assert event.source_module == "analyzer"

    def test_diagnosis_completed(self):
        event = AgentEvent.create("diagnosis_completed", "diagnosis_engine", {"failure_modes": ["no_parameter_change"]})
        assert event.event_type == "diagnosis_completed"
        assert event.source_module == "diagnosis_engine"

    def test_diagnosis_failed(self):
        event = AgentEvent.create("diagnosis_failed", "analyzer", {"error": "insufficient_data"})
        assert event.event_type == "diagnosis_failed"

    def test_diagnostic_design_selected(self):
        event = AgentEvent.create("diagnostic_design_selected", "diagnostic_runtime", {"design_id": "autograd_graph_audit_design"})
        assert event.event_type == "diagnostic_design_selected"

    def test_diagnostic_remote_execution_started(self):
        event = AgentEvent.create("diagnostic_remote_execution_started", "diagnostic_runtime", {"worker_id": "windows_wsl"})
        assert event.event_type == "diagnostic_remote_execution_started"

    def test_diagnostic_remote_execution_completed(self):
        event = AgentEvent.create("diagnostic_remote_execution_completed", "diagnostic_runtime", {"remote_job_id": "rj_123"})
        assert event.event_type == "diagnostic_remote_execution_completed"


class TestDiagnosticSourceModules:
    def test_analyzer_module(self):
        event = AgentEvent.create("diagnosis_completed", "analyzer", {})
        assert event.source_module == "analyzer"

    def test_diagnosis_engine_module(self):
        event = AgentEvent.create("diagnosis_started", "diagnosis_engine", {})
        assert event.source_module == "diagnosis_engine"

    def test_diagnostic_runtime_module(self):
        event = AgentEvent.create("diagnostic_remote_execution_completed", "diagnostic_runtime", {})
        assert event.source_module == "diagnostic_runtime"

    def test_existing_modules_still_work(self):
        event = AgentEvent.create("experiment_completed", "controller", {"status": "ok"})
        assert event.source_module == "controller"
