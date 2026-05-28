"""Tests for component probe report generation."""

import json
import tempfile
from pathlib import Path

from optiresearch.reports.component_probe_report import export_component_probe_report


class TestComponentProbeReport:
    def test_report_generation_creates_file(self, tmp_path):
        job_dir = tmp_path / "remote_job_aabbccdd11223344"
        job_dir.mkdir(parents=True)

        result = {
            "component": "fresnel",
            "surface_class": "Fresnel",
            "status": "succeeded",
            "differentiable": True,
            "parameters_changed": True,
            "trainable_param_count": 1,
            "params_with_grad": 1,
            "gradient_norm": 0.5,
            "loss_before": 1.0,
            "loss_after": 0.8,
            "evidence_level": "diagnostic_evidence",
            "claim_ceiling": "native_component_optimization",
            "trainable_param_names": ["f0"],
            "zero_gradient_parameters": [],
            "checked_component_candidates": ["fresnel", "binary2phase", "diffractive"],
            "caveats": ["Component-level evidence only"],
        }
        (job_dir / "result.json").write_text(json.dumps(result))

        metrics = {
            "component": "fresnel",
            "status": "succeeded",
        }
        (job_dir / "component_probe_metrics.json").write_text(json.dumps(metrics))

        job_result = {
            "job_id": "remote_job_aabbccdd11223344",
            "status": "succeeded",
            "remote_run_id": "run_xyz",
        }
        (job_dir / "remote_job_result.json").write_text(json.dumps(job_result))

        report_path = export_component_probe_report(
            "remote_job_aabbccdd11223344",
            remote_jobs_root=str(tmp_path),
        )
        assert report_path.exists()
        content = report_path.read_text()
        assert "Component Probe Report" in content
        assert "fresnel" in content
        assert "native_component_optimization" in content

    def test_report_handles_missing_result_file(self, tmp_path):
        job_dir = tmp_path / "remote_job_missing_data"
        job_dir.mkdir(parents=True)

        report_path = export_component_probe_report(
            "remote_job_missing_data",
            remote_jobs_root=str(tmp_path),
        )
        assert report_path.exists()
        content = report_path.read_text()
        assert "Component Probe Report" in content

    def test_report_includes_claim_boundaries(self, tmp_path):
        job_dir = tmp_path / "remote_job_boundary_01"
        job_dir.mkdir(parents=True)
        (job_dir / "result.json").write_text(json.dumps({
            "component": "fresnel",
            "claim_ceiling": "native_component_optimization",
            "evidence_level": "diagnostic_evidence",
        }))

        report_path = export_component_probe_report(
            "remote_job_boundary_01",
            remote_jobs_root=str(tmp_path),
        )
        content = report_path.read_text()
        assert "Claim Boundaries" in content
        assert "native_component_optimization" in content

    def test_report_includes_blocked_overclaims(self, tmp_path):
        job_dir = tmp_path / "remote_job_blocked_01"
        job_dir.mkdir(parents=True)
        (job_dir / "result.json").write_text(json.dumps({
            "component": "binary2phase",
            "claim_ceiling": "native_component_optimization",
        }))

        report_path = export_component_probe_report(
            "remote_job_blocked_01",
            remote_jobs_root=str(tmp_path),
        )
        content = report_path.read_text()
        assert "Blocked Overclaims" in content
        assert "full_geolens_direct_update" in content
        assert "hsi_improvement" in content
