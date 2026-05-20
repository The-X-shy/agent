from optiresearch.runtime.graph import run_mvp_flow


def test_run_mvp_flow_generates_trace_artifacts_memory_and_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    result = run_mvp_flow(
        "Design a mock depth-invariant and spectrally discriminative EDOF-HSI optical encoder",
        workspace_id="opti_lab",
    )

    assert result["run_id"]
    assert len(result["trace_ids"]) >= 1
    assert len(result["artifact_ids"]) >= 3
    assert result["experiment_spec"]["backend"] == "mock_deeplens"
    assert result["experiment_spec"]["optical_spec"]["encoder_type"] == "controlled_chromatic_edof"
    assert result["run_memory"]["run_id"] == result["run_id"]
    assert result["claims"]
    assert any(edge.get("metric_name") for claim in result["claims"] for edge in claim["support_edges"])
    assert result["context_pack"]["items"]
