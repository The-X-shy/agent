from datetime import datetime, timezone

from optiresearch.memory.schemas import MetaTrace, make_trace_id
from optiresearch.memory.skill_memory import SkillMemoryManager
from optiresearch.skills.router import SkillRouter
from optiresearch.storage.sqlite_store import SQLiteStore


def _skill_trace(status: str = "succeeded") -> MetaTrace:
    task = "run mock psf"
    return MetaTrace(
        trace_id=make_trace_id("ws", "run-1", "skill", "SimulationExperimentalist", task),
        workspace_id="ws",
        run_id="run-1",
        branch_id=None,
        step_id="skill",
        actor="SimulationExperimentalist",
        phase="Execute",
        task=task,
        skill_id="deeplens-adapter",
        skill_version="0.1.0",
        tool="MockDeepLensAdapter.simulate_psf_cube",
        input_refs=[],
        output_refs=["artifact-1"],
        findings=["artifact_types: metrics, psf_cube"],
        limitations=[] if status == "succeeded" else ["failed"],
        next_action=None,
        status=status,
        timestamp_start=datetime.now(timezone.utc),
        timestamp_end=datetime.now(timezone.utc),
        parents=[],
        content_hash=None,
        metadata={"command": "run_mock_psf", "artifact_types": ["metrics", "psf_cube"]},
    )


def test_update_from_trace_and_recommend_skills(tmp_path):
    store = SQLiteStore(tmp_path / "memory.sqlite")
    store.init_db()
    manager = SkillMemoryManager(store)

    memory = manager.update_from_trace(_skill_trace())
    recommendations = manager.recommend_skills("simulate psf", role="SimulationExperimentalist")

    assert memory.success_count == 1
    assert memory.success_rate == 1.0
    assert "metrics" in memory.emitted_artifact_types
    assert recommendations[0].skill_id == "deeplens-adapter"


def test_skill_router_uses_skill_memory_for_sorting(tmp_path):
    store = SQLiteStore(tmp_path / "memory.sqlite")
    store.init_db()
    manager = SkillMemoryManager(store)
    manager.update_from_trace(_skill_trace("succeeded"))

    resolved = SkillRouter(skill_memory_manager=manager).resolve(
        "SimulationExperimentalist",
        "simulate psf and inspect artifact metrics",
    )

    assert resolved[0].skill_id == "deeplens-adapter"
