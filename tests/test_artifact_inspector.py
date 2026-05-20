import numpy as np

from optiresearch.runtime.graph import run_mvp_flow
from optiresearch.skills.artifact_inspector.inspector import ArtifactInspector
from optiresearch.storage.file_artifact_store import FileArtifactStore
from optiresearch.storage.sqlite_store import SQLiteStore


def test_artifact_inspector_reads_metrics_json_and_npz(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    result = run_mvp_flow("Design a mock EDOF-HSI optical encoder")
    store = SQLiteStore(tmp_path / "memory.sqlite")
    artifact_store = FileArtifactStore(root=tmp_path / "artifacts", store=store)
    inspector = ArtifactInspector(artifact_store)
    artifacts = artifact_store.list_artifacts(run_id=result["run_id"])

    metrics_ref = next(item for item in artifacts if item.metadata["artifact_type"] == "metrics")
    npz_ref = next(item for item in artifacts if item.metadata["artifact_type"] == "psf_cube")
    metrics_summary = inspector.inspect_artifact(metrics_ref)
    npz_summary = inspector.inspect_artifact(npz_ref)

    assert "psf_depth_similarity" in metrics_summary["metric_names"]
    assert metrics_ref.metadata["metric_names"]
    assert npz_summary["arrays"]["psf_cube"] == [9, 31, 32, 32]


def test_artifact_inspector_handles_real_backend_psf_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    source = tmp_path / "real_psf_cube.npz"
    np.savez_compressed(source, psf_cube=np.zeros((2, 3, 8, 8), dtype=np.float32))
    store = SQLiteStore(tmp_path / "memory.sqlite")
    artifact_store = FileArtifactStore(root=tmp_path / "artifacts", store=store)
    ref = artifact_store.register_file(
        source,
        workspace_id="default",
        run_id="run_real",
        trace_id=None,
        producer="DeepLensAdapter.simulate_psf_cube",
        metadata={"filename": "psf_cube.npz", "backend": "deeplens"},
        metrics={},
    )

    summary = ArtifactInspector(artifact_store).inspect_artifact(ref)

    assert summary["artifact_type"] == "psf_cube"
    assert summary["arrays"]["psf_cube"] == [2, 3, 8, 8]
