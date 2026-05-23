"""Phase 34: Execution fidelity report section tests."""

import json
import tempfile
from pathlib import Path

from optiresearch.reports.autonomous_loop_report import _build_execution_fidelity
from optiresearch.reports.remote_execution import _markdown as remote_execution_markdown
from optiresearch.reports.native_geolens_hsi_report import (
    export_native_geolens_hsi_report,
    _markdown as native_geolens_hsi_markdown,
)


class _FakeIteration:
    def __init__(self, iteration_id, backend_id, payload):
        self.iteration_id = iteration_id
        self.execution_result = {
            "backend_id": backend_id,
            "result_payload": payload,
        }
        self.strategy_recommendation = None
        self.claim_gate_decision = None
        self.metrics_snapshot = None
        self.experiment_spec = None


class _FakeResult:
    def __init__(self, iterations):
        self.iterations = iterations


def test_autonomous_loop_report_has_all_fidelity_columns():
    """All 7 execution fidelity columns must appear in the table header."""
    payload = {
        "execution_fidelity": "deeplens_native_geometric",
        "proxy_fallback_used": False,
        "deeplens_native_psf_path": "geolens.psf_geometric",
        "full_wave_optics": False,
        "phase_to_fft_proxy_used": False,
        "platform": "Linux",
    }
    fake = _FakeResult([_FakeIteration(1, "deeplens_geolens_geometric", payload)])
    output = _build_execution_fidelity(fake)

    assert "Execution Fidelity" in output
    assert "Full Wave Optics" in output
    assert "Phase-to-FFT Proxy" in output
    assert "Platform" in output
    assert "Proxy Fallback" in output
    assert "Native PSF Path" in output
    assert "deeplens_native_geometric" in output
    assert "geolens.psf_geometric" in output


def test_remote_execution_report_has_fidelity_section():
    """Remote execution report must include Execution Fidelity section."""
    result = {
        "status": "succeeded",
        "remote_run_id": "test-run-1",
        "error_code": None,
        "execution_fidelity": "deeplens_native_geometric",
        "proxy_fallback_used": False,
        "deeplens_native_psf_path": "geolens.psf_geometric",
        "full_wave_optics": False,
        "phase_to_fft_proxy_used": False,
        "platform": "Linux",
        "metrics_summary": {
            "fallback_used": False,
            "execution_fidelity": "deeplens_native_geometric",
        },
        "caveats": [],
    }
    ingestion = {
        "artifact_ids": [],
        "claims": [],
    }
    output = remote_execution_markdown("test-job", result, ingestion)

    assert "Execution Fidelity" in output
    assert "deeplens_native_geometric" in output
    assert "proxy_fallback_used" in output
    assert "deeplens_native_psf_path" in output
    assert "full_wave_optics" in output
    assert "phase_to_fft_proxy_used" in output
    assert "platform" in output


def test_native_geolens_hsi_report_has_fidelity_section():
    """Native GeoLens HSI report must include Execution Fidelity section."""
    spec = {
        "candidate": "GeoLensCooke",
        "reconstructor": "differentiable_linear",
    }
    result = {
        "status": "succeeded",
        "candidate": "GeoLensCooke",
        "reconstructor": "differentiable_linear",
        "execution_fidelity": "deeplens_native_geometric",
        "proxy_fallback_used": False,
        "deeplens_native_psf_path": "geolens.psf_geometric",
        "full_wave_optics": False,
        "phase_to_fft_proxy_used": False,
        "platform": "Linux",
        "reconstruction_loss_before": 0.5,
        "reconstruction_loss_after": 0.3,
        "optical_gradient_norm": 0.01,
        "optical_parameters_changed": True,
        "rollback_count": 1,
        "accepted_update_count": 3,
        "rejected_update_count": 1,
        "stable_training_succeeded": True,
        "evidence_level": "native_lens_simulation",
        "caveats": [],
    }
    output = native_geolens_hsi_markdown("test-run", spec, result)

    assert "Execution Fidelity" in output
    assert "deeplens_native_geometric" in output
    assert "proxy_fallback_used" in output
    assert "geolens.psf_geometric" in output
    assert "Metrics" in output
    assert "Optical Parameters" in output
    assert "Caveats" in output


def test_fidelity_section_handles_missing_data():
    """When fidelity fields are absent, the section should handle gracefully."""
    spec = {"candidate": "GeoLensCooke"}
    result = {"status": "unsupported", "error_code": "GEOLENS_PSF_GEOMETRIC_FAILED_INDEXERROR"}
    output = native_geolens_hsi_markdown("test-run-2", spec, result)

    assert "Execution Fidelity" in output
    # Should still render the section even with minimal data
    assert "execution_fidelity" in output


def test_native_geolens_hsi_report_export_writes_file():
    """Export function should write a markdown file."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "native_geolens_hsi" / "test-run-export"
        run_dir.mkdir(parents=True)
        (run_dir / "spec.json").write_text(json.dumps({
            "candidate": "GeoLensCooke",
            "reconstructor": "differentiable_linear",
        }))
        (run_dir / "result.json").write_text(json.dumps({
            "status": "succeeded",
            "execution_fidelity": "deeplens_native_geometric",
            "proxy_fallback_used": False,
            "deeplens_native_psf_path": "geolens.psf_geometric",
            "caveats": [],
        }))
        path = export_native_geolens_hsi_report("test-run-export", output_root=tmp)
        assert path.exists()
        content = path.read_text()
        assert "Native GeoLens HSI Report" in content
        assert "Execution Fidelity" in content


def test_empty_iterations_generate_placeholder():
    """When no iterations have fidelity data, a placeholder message appears."""
    fake = _FakeResult([])
    output = _build_execution_fidelity(fake)
    assert "No execution fidelity data" in output
