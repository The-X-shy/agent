from optiresearch.schemas.autonomous import AutonomousLoopConfig
from optiresearch.runtime.autonomous_loop import run_autonomous_research_loop


def test_autonomous_loop_remote_mode_uses_remote_executor(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    monkeypatch.setenv("OPTIRESEARCH_HSI_ROOT", str(tmp_path / "hsi"))
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    calls = []

    def fake_remote_execution(*, worker_id, objective, backend, encoder, reconstructor, dataset):
        calls.append(
            {
                "worker_id": worker_id,
                "objective": objective,
                "backend": backend,
                "encoder": encoder,
                "reconstructor": reconstructor,
                "dataset": dataset,
            }
        )
        return {
            "status": "succeeded",
            "run_id": "remote_run_1",
            "metrics": {"reconstruction_score": 0.7, "PSNR": 30.0},
            "artifact_ids": [],
            "artifact_uris": [],
            "evidence_level": "remote_deeplens_worker",
        }

    config = AutonomousLoopConfig(
        objective="Remote autonomous test",
        max_iterations=1,
        llm_provider="mock",
        backend="deeplens",
        dataset="synthetic",
        allowed_encoders=["controlled_chromatic_edof"],
        allowed_reconstructors=["optical_conditioned_linear"],
        metadata={"execution_mode": "remote", "worker_id": "windows_wsl"},
    )

    summary = run_autonomous_research_loop(config, remote_executor=fake_remote_execution)

    assert calls
    assert calls[0]["worker_id"] == "windows_wsl"
    assert summary.iterations[0].run_id == "remote_run_1"
