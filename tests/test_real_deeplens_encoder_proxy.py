import json
import os

import pytest

from optiresearch.runtime.baselines import ENCODER_TYPES, run_baseline_batch


@pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS") != "1",
    reason="Real DeepLens proxy baseline requires explicit opt-in.",
)
def test_real_deeplens_encoder_proxy_baseline_generates_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_DB_PATH", str(tmp_path / "memory.sqlite"))
    monkeypatch.setenv("OPTIRESEARCH_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    report = run_baseline_batch("Real DeepLens encoder proxy", backend="deeplens", output_root=tmp_path / "deeplens")

    assert [item["encoder_type"] for item in report["runs"]] == ENCODER_TYPES
    assert all(item["joint_tradeoff_score"] is not None for item in report["runs"])
    assert all(item["metrics"]["encoder_behavior_realization_level"] == "adapter_proxy" for item in report["runs"])
    manifests = list((tmp_path / "artifacts").glob("**/proxy_transform_manifest.json"))
    assert manifests
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["realization_level"] == "adapter_proxy"
