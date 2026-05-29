"""CLI alias tests for native GeoLens geometric HSI co-design."""

from types import SimpleNamespace


def test_local_native_geolens_geometric_alias_maps_steps(monkeypatch, capsys):
    import optiresearch.cli as cli
    import optiresearch.runtime.stable_native_lens_hsi_loop as loop

    captured = {}

    def fake_run(spec):
        captured["spec"] = spec
        return SimpleNamespace(
            status="succeeded",
            parameter_count=14,
            trainable_param_count=14,
            params_with_grad=14,
            grad_norm_max=0.1,
            psf_requires_grad=True,
            loss_requires_grad=True,
            graph_connected=True,
            optical_parameters_changed=True,
            mse_before=1.0,
            mse_after=0.9,
            psnr_before=10.0,
            psnr_after=10.5,
            sam_before=0.2,
            sam_after=0.1,
            deeplens_native_psf_path="geolens.psf_geometric",
            evidence_level="native_lens_simulation",
            error_code=None,
        )

    monkeypatch.setattr(loop, "run_stable_native_lens_hsi_codesign", fake_run)

    cli.main([
        "run-native-geolens-geometric-hsi-codesign",
        "--dataset", "synthetic",
        "--steps", "4",
        "--device", "cpu",
    ])

    output = capsys.readouterr().out
    assert '"status": "succeeded"' in output
    assert '"parameter_count": 14' in output
    assert captured["spec"].max_steps == 4
    assert captured["spec"].candidate == "GeoLensCooke"
    assert captured["spec"].full_wave_optics is False


def test_remote_native_geolens_geometric_alias_maps_steps(monkeypatch, capsys):
    import optiresearch.cli as cli

    captured = {}

    def fake_remote(worker_id, **kwargs):
        captured["worker_id"] = worker_id
        captured.update(kwargs)
        return {"result": {"status": "succeeded", "job_id": "remote_job_alias"}}

    monkeypatch.setattr(cli, "run_remote_deeplens_native_geolens_hsi_codesign", fake_remote)

    cli.main([
        "run-remote-native-geolens-geometric-hsi-codesign",
        "--worker-id", "windows_wsl",
        "--steps", "5",
        "--device", "cpu",
    ])

    output = capsys.readouterr().out
    assert '"job_id": "remote_job_alias"' in output
    assert captured["worker_id"] == "windows_wsl"
    assert captured["max_steps"] == 5
    assert captured["lens_file"] == "auto:cooke"
