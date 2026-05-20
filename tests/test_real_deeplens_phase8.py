import os

import pytest

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.adapters.deeplens_api_probe import probe_deeplens_api
from optiresearch.runtime.baselines import run_baseline_batch
from optiresearch.schemas.optimization import build_default_optimization_spec


@pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS") != "1",
    reason="Real DeepLens Phase 8 test requires explicit opt-in.",
)
def test_real_deeplens_phase8_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    probe = probe_deeplens_api()
    report = run_baseline_batch("phase8 real", backend="deeplens", realization="auto", output_root=tmp_path / "baselines")
    optimization = DeepLensAdapter().run_optimization(build_default_optimization_spec(["joint_score"]), None)

    assert probe["available"] is True
    assert report["runs"]
    assert report["runs"][0]["metrics"]["selected_realization_level"] in {"semi_native", "adapter_proxy"}
    assert list((tmp_path / "artifacts").glob("**/realization_manifest.json"))
    assert optimization.errors[0]["code"] == "OPTIMIZATION_NOT_AVAILABLE"
