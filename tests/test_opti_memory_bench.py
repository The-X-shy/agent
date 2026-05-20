from optiresearch.benchmarks.opti_memory_bench.runner import OptiMemoryBenchRunner
from optiresearch.storage.sqlite_store import SQLiteStore


def test_opti_memory_bench_runs_three_tasks_and_writes_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    runner = OptiMemoryBenchRunner(
        store=SQLiteStore(tmp_path / "memory.sqlite"),
        report_root=tmp_path / "benchmarks",
    )

    report = runner.run()

    assert {item["task_type"] for item in report["tasks"]} == {
        "DeepLens-Recipe-Reuse",
        "EDOF-HSI-Claim-QA",
        "Skill-Load-Efficiency",
    }
    assert report["summary"]["task_count"] == 3
    assert (tmp_path / "benchmarks" / "opti_memory_bench_report.json").exists()
    assert (tmp_path / "benchmarks" / "opti_memory_bench_report.md").exists()


def test_opti_memory_bench_ablation_modes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    runner = OptiMemoryBenchRunner(
        store=SQLiteStore(tmp_path / "memory.sqlite"),
        report_root=tmp_path / "benchmarks",
    )

    report = runner.run(mode="full_rmos")
    ablations = report["ablations"]

    assert set(ablations) == {"no_memory", "trace_only", "plan_only", "skill_only", "full_rmos"}
    assert ablations["full_rmos"]["total_score"] >= ablations["no_memory"]["total_score"]
    for mode_metrics in ablations.values():
        assert {"plan_hit", "evidence_complete", "unsupported_claim_rate", "trigger_precision", "total_score"} <= set(mode_metrics)
