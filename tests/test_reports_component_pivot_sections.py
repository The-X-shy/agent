"""Test report sections for GeoLens autograd findings and component pivot."""

import json
from pathlib import Path


class TestGeoLensDiagnosticFindingSection:
    def test_autograd_audit_metrics_have_all_required_fields(self):
        """Verify the Phase 60 autograd audit result has the fields needed for reports."""
        result_path = Path("workspace/remote_jobs/remote_job_53f5e98e37bdeed0/remote_job_53f5e98e37bdeed0/result.json")
        if not result_path.exists():
            import pytest
            pytest.skip("Remote autograd audit result not available locally")

        data = json.loads(result_path.read_text(encoding="utf-8"))
        assert "graph_connected" in data
        assert "psf_requires_grad" in data
        assert "loss_requires_grad" in data
        assert "trainable_param_count" in data
        assert "parameter_count" in data or "params_with_grad" in data

    def test_remote_diagnostic_report_has_lens_section(self):
        """Verify remote diagnostic reports include Lens Resolution section."""
        report_path = Path("workspace/remote_jobs/remote_job_53f5e98e37bdeed0/remote_diagnostic_report.md")
        if not report_path.exists():
            import pytest
            pytest.skip("Remote diagnostic report not available locally")

        content = report_path.read_text(encoding="utf-8")
        assert "Lens Resolution" in content
        assert "Diagnostic Results" in content
        assert "Gradient Flow Interpretation" in content

    def test_agent_plan_report_has_diagnosis_section(self):
        """Verify agent plan execution reports include diagnosis information."""
        import glob
        reports = sorted(Path("workspace/agent_plan_executions").glob("*/plan_execution_report.md"))
        if not reports:
            import pytest
            pytest.skip("No agent plan execution reports available")

        content = reports[-1].read_text(encoding="utf-8")
        assert "Seed Evidence" in content or "Diagnosis" in content
        assert "Strategy" in content or "Design" in content
