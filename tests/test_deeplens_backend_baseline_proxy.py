from optiresearch.adapters.base import AdapterMetricBundle, AdapterRunResult
from optiresearch.runtime.baselines import run_baseline_batch


def test_deeplens_baseline_report_includes_proxy_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    def fake_execute(self, skill_id, command, args=None):
        spec = args["spec"]
        encoder = spec.optical_spec.encoder_type
        scores = {
            "conventional": (0.45, 0.08, 0.72, 0.91),
            "achromatic": (0.76, 0.07, 0.82, 0.88),
            "edof": (0.88, 0.15, 0.58, 0.8),
            "chromatic_coded": (0.52, 0.7, 0.48, 0.73),
            "controlled_chromatic_edof": (0.86, 0.62, 0.66, 0.84),
        }
        depth, spectral, mtf, energy = scores[encoder]
        output_dir = tmp_path / "raw" / encoder
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "optical_metrics.json"
        metrics_path.write_text(
            '{"psf_depth_similarity": %s, "spectral_separability": %s}' % (depth, spectral),
            encoding="utf-8",
        )
        return {
            "status": "succeeded",
            "artifacts": [str(metrics_path)],
            "metrics": {
                "encoder_type": encoder,
                "depth_planes": 9,
                "wavelength_bands": 31,
                "psf_depth_similarity": depth,
                "spectral_separability": spectral,
                "mock_mtf_mean": mtf,
                "mock_energy_efficiency": energy,
                "deeplens_mtf_mean": mtf,
                "deeplens_energy_efficiency": energy,
                "backend_capability_level": "proxy",
                "encoder_behavior_realized": True,
                "encoder_behavior_realization_level": "adapter_proxy",
                "physical_validation_level": "deeplens_base_psf_plus_adapter_proxy",
                "proxy_transform_name": f"{encoder}_proxy_transform",
            },
            "logs": [],
            "errors": [],
            "metadata": {
                "backend": "deeplens",
                "backend_capability_level": "proxy",
                "encoder_behavior_realized": True,
                "encoder_behavior_realization_level": "adapter_proxy",
                "physical_validation_level": "deeplens_base_psf_plus_adapter_proxy",
                "proxy_transform_name": f"{encoder}_proxy_transform",
            },
        }

    monkeypatch.setattr("optiresearch.skills.executor.SkillExecutor.execute", fake_execute)

    report = run_baseline_batch("Design encoder proxy baselines", backend="deeplens", output_root=tmp_path / "deeplens")
    markdown = (tmp_path / "deeplens" / "baseline_comparison.md").read_text(encoding="utf-8")

    assert len({item["joint_tradeoff_score"] for item in report["runs"]}) > 1
    assert report["best_joint_tradeoff"]["encoder_type"] == "controlled_chromatic_edof"
    assert "Realization Level" in markdown
    assert "Physical Validation" in markdown
    assert "adapter_proxy" in markdown
