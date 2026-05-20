import json

from optiresearch.reports.phase7 import export_phase7_report


def test_phase7_report_exports_proxy_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_BASELINE_ROOT", str(tmp_path / "baselines"))
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    mock_dir = tmp_path / "baselines" / "mock_deeplens"
    real_dir = tmp_path / "baselines" / "deeplens"
    mock_dir.mkdir(parents=True)
    real_dir.mkdir(parents=True)
    mock_payload = {
        "backend": "mock_deeplens",
        "objective": "phase7",
        "runs": [
            {"encoder_type": "controlled_chromatic_edof", "metrics": {"backend_capability_level": "mock", "encoder_behavior_realized": True}, "joint_tradeoff_score": 0.75}
        ],
        "best_joint_tradeoff": {"encoder_type": "controlled_chromatic_edof", "joint_tradeoff_score": 0.75},
    }
    real_payload = {
        "backend": "deeplens",
        "objective": "phase7",
        "runs": [
            {
                "encoder_type": "controlled_chromatic_edof",
                "metrics": {
                    "backend_capability_level": "proxy",
                    "encoder_behavior_realized": True,
                    "encoder_behavior_realization_level": "adapter_proxy",
                    "physical_validation_level": "deeplens_base_psf_plus_adapter_proxy",
                    "proxy_transform_name": "controlled_chromatic_edof_proxy_transform",
                },
                "joint_tradeoff_score": 0.7,
            }
        ],
        "best_joint_tradeoff": {"encoder_type": "controlled_chromatic_edof", "joint_tradeoff_score": 0.7},
    }
    (mock_dir / "baseline_comparison.json").write_text(json.dumps(mock_payload), encoding="utf-8")
    (real_dir / "baseline_comparison.json").write_text(json.dumps(real_payload), encoding="utf-8")

    path = export_phase7_report()

    text = path.read_text(encoding="utf-8")
    assert path.name == "phase7_deeplens_encoder_proxy_report.md"
    assert "Encoder strategy registry" in text
    assert "adapter_proxy" in text
    assert "not native physical encoder optimization" in text
