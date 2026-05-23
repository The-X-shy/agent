"""Phase 31: Autonomous loop report backend probe tests."""

import tempfile
from pathlib import Path

from optiresearch.reports.autonomous_loop_report import (
    export_autonomous_loop_report,
)
from optiresearch.schemas.autonomous_loop import (
    AutonomousLoopResult,
    AutonomousLoopIteration,
)


def test_report_includes_backend_probe_results():
    iterations = [
        AutonomousLoopIteration(
            iteration_id=1,
            execution_result={
                "backend_id": "phase_to_fft_proxy",
                "result_payload": {
                    "reconstruction_loss_after": 0.001,
                    "probe_status": "succeeded",
                    "probe_time_seconds": 0.5,
                },
            },
        ),
    ]
    result = AutonomousLoopResult(
        loop_id="test_probe_report",
        status="completed",
        objective="test",
        iterations=iterations,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_autonomous_loop_report(result, Path(tmpdir))
        content = path.read_text()
        assert "Backend Probe Results" in content


def test_report_includes_backend_switch_validation():
    iterations = [
        AutonomousLoopIteration(
            iteration_id=1,
            execution_result={
                "backend_id": "phase_to_fft_proxy",
                "switched_from_backend": "phase_to_fft_proxy",
                "switched_to_backend": "deeplens_geolens_geometric",
                "backend_switch_validated": True,
                "result_payload": {"probe_status": "succeeded"},
            },
        ),
    ]
    result = AutonomousLoopResult(
        loop_id="test_validation_report",
        status="completed",
        objective="test",
        iterations=iterations,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_autonomous_loop_report(result, Path(tmpdir))
        content = path.read_text()
        assert "Backend Switch Validation" in content
        assert "Switch Triggered:" in content
        assert "Switch Validated:" in content
        assert "Probe Success:" in content
