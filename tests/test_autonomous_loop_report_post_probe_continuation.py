"""Phase 32: Autonomous loop report post-probe continuation tests."""

import tempfile
from pathlib import Path

from optiresearch.reports.autonomous_loop_report import export_autonomous_loop_report
from optiresearch.schemas.autonomous_loop import (
    AutonomousLoopResult, AutonomousLoopIteration,
)


def test_report_includes_post_probe_continuation():
    iterations = [
        AutonomousLoopIteration(
            iteration_id=1,
            execution_result={
                "backend_id": "deeplens_geolens_geometric",
                "post_probe_continuation_required": True,
                "validated_backend_id": "deeplens_geolens_geometric",
                "validated_backend_evidence_level": "native_lens_simulation",
                "result_payload": {"status": "succeeded", "reconstruction_loss_after": 0.01},
            },
        ),
    ]
    result = AutonomousLoopResult(
        loop_id="test_continuation_report",
        status="completed",
        objective="test",
        iterations=iterations,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_autonomous_loop_report(result, Path(tmpdir))
        content = path.read_text()
        assert "Post-Probe Continuation" in content


def test_report_includes_alternative_backend_attempts():
    iterations = [
        AutonomousLoopIteration(
            iteration_id=1,
            execution_result={
                "backend_id": "phase_to_fft_proxy",
                "alternative_backends_attempted": [
                    "deeplens_geolens_geometric", "deeplens_fresnel_component",
                ],
            },
        ),
    ]
    result = AutonomousLoopResult(
        loop_id="test_alternatives_report",
        status="completed",
        objective="test",
        iterations=iterations,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_autonomous_loop_report(result, Path(tmpdir))
        content = path.read_text()
        assert "Alternative Backend Attempts" in content
