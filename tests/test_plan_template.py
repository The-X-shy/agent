from optiresearch.memory.plan_template import PlanTemplateManager
from optiresearch.runtime.graph import run_mvp_flow
from optiresearch.storage.sqlite_store import SQLiteStore


def test_create_default_templates_and_match(tmp_path):
    store = SQLiteStore(tmp_path / "memory.sqlite")
    store.init_db()
    manager = PlanTemplateManager(store)
    manager.create_default_templates()

    matches = manager.match("evaluate edof hsi")

    assert {item.template_id for item in matches} & {"evaluate_edof_hsi_encoder", "evaluate_mock_optical_encoder"}


def test_compile_from_run_updates_success_rate(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    result = run_mvp_flow("Design a mock EDOF-HSI optical encoder")
    manager = PlanTemplateManager(SQLiteStore(tmp_path / "memory.sqlite"))

    template = manager.compile_from_run(result["run_id"])

    assert template.template_id == "evaluate_mock_optical_encoder"
    assert template.historical_success_rate == 1.0
    assert "deeplens-adapter" in template.metadata["used_skill_ids"]
    assert "metrics" in template.metadata["artifact_types"]
