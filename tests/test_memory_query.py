from optiresearch.memory.router import MemoryRouter
from optiresearch.runtime.graph import run_mvp_flow
from optiresearch.storage.file_artifact_store import FileArtifactStore
from optiresearch.storage.sqlite_store import SQLiteStore


def test_memory_router_returns_expected_context_by_intent(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.sqlite"
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(db_path))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(artifact_root))
    run_mvp_flow("Design a mock EDOF-HSI optical encoder", workspace_id="opti_lab")
    store = SQLiteStore(db_path)
    artifacts = FileArtifactStore(root=artifact_root, store=store)
    router = MemoryRouter(store=store, artifact_store=artifacts)

    evidence = router.query("CriticalReviewer", "evidence claim", "depth stability", scope={})
    plan = router.query("LeadInvestigator", "plan", "mock optical encoder", scope={})
    default = router.query("System", "status", "mock optical encoder", scope={})

    assert {item["type"] for item in evidence["items"]} & {"claim", "artifact", "trace"}
    assert any(item["type"] == "claim" and item["evidence_refs"] for item in evidence["items"])
    assert {item["type"] for item in plan["items"]} & {"run_memory", "plan_template"}
    assert {item["type"] for item in default["items"]} & {"run_memory", "trace"}


def test_memory_router_returns_plan_and_skill_memory_items(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.sqlite"
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(db_path))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(artifact_root))
    run_mvp_flow("Design a mock EDOF-HSI optical encoder", workspace_id="opti_lab")
    store = SQLiteStore(db_path)
    artifacts = FileArtifactStore(root=artifact_root, store=store)
    router = MemoryRouter(store=store, artifact_store=artifacts)

    plan = router.query("LeadInvestigator", "plan", "evaluate edof hsi", scope={})
    skill = router.query("SimulationExperimentalist", "skill", "simulate psf", scope={})

    assert any(item["type"] == "plan_template" for item in plan["items"])
    assert any(item["type"] == "skill_memory" for item in skill["items"])
    assert all({"type", "id", "summary", "score", "evidence_refs", "source"} <= set(item) for item in skill["items"])
