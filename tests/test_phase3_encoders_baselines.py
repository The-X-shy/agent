import json

from optiresearch.adapters.mock_deeplens import MockDeepLensAdapter
from optiresearch.runtime.baselines import ENCODER_TYPES, run_baseline_batch
from optiresearch.schemas.experiment import build_default_mock_edof_hsi_experiment


def test_mock_deeplens_metrics_vary_by_encoder_type(tmp_path):
    adapter = MockDeepLensAdapter(seed=42)
    metrics_by_encoder = {}
    for encoder_type in ENCODER_TYPES:
        experiment = build_default_mock_edof_hsi_experiment("compare encoders", encoder_type=encoder_type)
        result = adapter.simulate_psf_cube(experiment, None, tmp_path / encoder_type)
        metrics_by_encoder[encoder_type] = result["metrics"]

    assert metrics_by_encoder["controlled_chromatic_edof"]["psf_depth_similarity"] > metrics_by_encoder["conventional"]["psf_depth_similarity"]
    assert metrics_by_encoder["controlled_chromatic_edof"]["spectral_separability"] > metrics_by_encoder["achromatic"]["spectral_separability"]
    assert metrics_by_encoder["achromatic"]["mock_mtf_mean"] >= metrics_by_encoder["chromatic_coded"]["mock_mtf_mean"]
    assert len({tuple(item.values()) for item in metrics_by_encoder.values()}) == 5


def test_run_baseline_batch_writes_comparison_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    report = run_baseline_batch(
        "Design depth-invariant and spectrally discriminative EDOF-HSI encoder",
        workspace_id="baseline_test",
        output_root=tmp_path / "baselines",
    )

    assert [item["encoder_type"] for item in report["runs"]] == ENCODER_TYPES
    assert report["best_joint_tradeoff"]["encoder_type"] == "controlled_chromatic_edof"
    assert (tmp_path / "baselines" / "baseline_comparison.json").exists()
    assert (tmp_path / "baselines" / "baseline_comparison.md").exists()
    payload = json.loads((tmp_path / "baselines" / "baseline_comparison.json").read_text(encoding="utf-8"))
    assert payload["best_joint_tradeoff"]["encoder_type"] == "controlled_chromatic_edof"
