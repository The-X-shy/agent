from optiresearch.runtime.baselines import run_baseline_batch


def test_run_baseline_batch_accepts_realization_modes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    def fake_execute(self, skill_id, command, args=None):
        encoder = args["spec"].optical_spec.encoder_type
        realization = args.get("realization", "auto")
        selected = "semi_native" if encoder == "conventional" and realization in {"auto", "semi_native"} else "adapter_proxy"
        output = tmp_path / "raw" / encoder
        output.mkdir(parents=True, exist_ok=True)
        metrics_path = output / "optical_metrics.json"
        metrics_path.write_text('{"psf_depth_similarity": 0.5}', encoding="utf-8")
        return {
            "status": "succeeded",
            "artifacts": [str(metrics_path)],
            "metrics": {
                "encoder_type": encoder,
                "depth_planes": 9,
                "wavelength_bands": 31,
                "psf_depth_similarity": 0.5,
                "spectral_separability": 0.4,
                "mock_mtf_mean": 0.5,
                "mock_energy_efficiency": 0.8,
                "deeplens_mtf_mean": 0.5,
                "deeplens_energy_efficiency": 0.8,
                "backend_capability_level": selected,
                "encoder_behavior_realized": True,
                "selected_realization_level": selected,
                "encoder_behavior_realization_level": selected,
                "physical_validation_level": "baseline DeepLens ParaxialLens behavior" if selected == "semi_native" else "deeplens_base_psf_plus_adapter_proxy",
                "semi_native_attempted": realization == "semi_native",
                "semi_native_succeeded": selected == "semi_native",
                "proxy_fallback_used": selected != "semi_native",
                "claim_scope": "baseline DeepLens ParaxialLens behavior",
            },
            "metadata": {
                "backend": "deeplens",
                "backend_capability_level": selected,
                "encoder_behavior_realized": True,
                "selected_realization_level": selected,
            },
            "logs": [],
            "errors": [],
        }

    monkeypatch.setattr("optiresearch.skills.executor.SkillExecutor.execute", fake_execute)

    report = run_baseline_batch("realization mode", backend="deeplens", realization="semi_native", output_root=tmp_path / "baselines")

    assert report["realization"] == "semi_native"
    assert report["runs"][0]["metrics"]["selected_realization_level"] == "semi_native"
    assert "Claim Scope" in (tmp_path / "baselines" / "baseline_comparison.md").read_text(encoding="utf-8")
