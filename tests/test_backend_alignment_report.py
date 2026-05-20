import json

from optiresearch.reports.backend_alignment import (
    compare_backend_metrics,
    export_backend_alignment_report,
)


def test_backend_alignment_report_exports_caveat(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_BASELINE_ROOT", str(tmp_path / "baselines"))
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(tmp_path / "reports"))
    mock_dir = tmp_path / "baselines" / "mock_deeplens"
    real_dir = tmp_path / "baselines" / "deeplens"
    mock_dir.mkdir(parents=True)
    real_dir.mkdir(parents=True)
    mock_payload = {
        "backend": "mock_deeplens",
        "runs": [
            {
                "encoder_type": "controlled_chromatic_edof",
                "metrics": {
                    "depth_planes": 9,
                    "wavelength_bands": 31,
                    "psf_depth_similarity": 0.88,
                    "spectral_separability": 0.64,
                    "mock_mtf_mean": 0.66,
                    "mock_energy_efficiency": 0.84,
                    "encoder_behavior_realized": True,
                    "backend_capability_level": "mock",
                },
                "joint_tradeoff_score": 0.757,
            }
        ],
    }
    real_payload = {
        "backend": "deeplens",
        "runs": [
            {
                "encoder_type": "controlled_chromatic_edof",
                "metrics": {
                    "depth_planes": 9,
                    "wavelength_bands": 31,
                    "psf_depth_similarity": 0.56,
                    "spectral_separability": 0.0,
                    "deeplens_mtf_mean": 0.0,
                    "deeplens_energy_efficiency": 0.89,
                    "encoder_behavior_realized": True,
                    "backend_capability_level": "proxy",
                    "encoder_behavior_realization_level": "adapter_proxy",
                    "physical_validation_level": "deeplens_base_psf_plus_adapter_proxy",
                    "proxy_transform_name": "controlled_chromatic_edof_proxy_transform",
                },
                "joint_tradeoff_score": 0.69,
            }
        ],
    }
    (mock_dir / "baseline_comparison.json").write_text(json.dumps(mock_payload), encoding="utf-8")
    (real_dir / "baseline_comparison.json").write_text(json.dumps(real_payload), encoding="utf-8")

    comparison = compare_backend_metrics("mock_deeplens", "deeplens")
    paths = export_backend_alignment_report("mock_deeplens", "deeplens")

    assert comparison["rows"][0]["encoder_type"] == "controlled_chromatic_edof"
    assert comparison["summary"]["rank_agreement"] == 1.0
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "Proxy Realization" in text
    assert "Native vs Proxy Distinction" in text
    assert "Claims Allowed / Not Allowed" in text
    assert paths["json"].exists()
