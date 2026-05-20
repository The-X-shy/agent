from optiresearch.runtime.graph import run_mvp_flow
from optiresearch.storage.file_artifact_store import FileArtifactStore
from optiresearch.storage.sqlite_store import SQLiteStore


def test_mock_backend_metadata_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    result = run_mvp_flow("Design a mock EDOF-HSI encoder", backend="mock_deeplens")
    store = SQLiteStore(tmp_path / "memory.sqlite")
    artifacts = FileArtifactStore(root=tmp_path / "artifacts", store=store).list_artifacts(result["run_id"])
    traces = store.list("meta_traces", run_id=result["run_id"])
    run_memory = result["run_memory"]
    claims = result["claims"]

    assert run_memory["metadata"]["backend"] == "mock_deeplens"
    assert run_memory["metadata"]["backend_capability_level"] == "mock"
    assert all(trace["metadata"].get("backend") == "mock_deeplens" for trace in traces)
    assert all(artifact.metadata["backend"] == "mock_deeplens" for artifact in artifacts)
    assert all(claim["metadata"]["backend"] == "mock_deeplens" for claim in claims)


def test_deeplens_unavailable_metadata_is_structured(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    result = run_mvp_flow("Design a minimal DeepLens PSF smoke run", backend="deeplens")
    store = SQLiteStore(tmp_path / "memory.sqlite")
    traces = store.list("meta_traces", run_id=result["run_id"])

    assert result["run_memory"]["metadata"]["backend"] == "deeplens"
    assert result["run_memory"]["metadata"]["backend_capability_level"] in {"smoke", "minimal", "proxy"}
    # encoder_behavior_realized depends on DeepLens availability — not asserted unconditionally
    assert all(trace["metadata"].get("backend") == "deeplens" for trace in traces)
    if result["errors"] and len(result["errors"]) > 0:
        error_code = result["errors"][0].get("code") if isinstance(result["errors"][0], dict) else ""
        assert error_code in {"DEEPLENS_NOT_INSTALLED", "DEEPLENS_API_UNSUPPORTED", ""}
