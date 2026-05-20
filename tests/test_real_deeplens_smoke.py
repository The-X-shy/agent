import os

import pytest

from optiresearch.agents.method_builder import MethodBuilder
from optiresearch.adapters.deeplens import DeepLensAdapter


@pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS") != "1",
    reason="Real DeepLens smoke tests are opt-in.",
)
def test_real_deeplens_smoke_outputs_artifacts(tmp_path):
    spec = MethodBuilder().build_mock_optical_spec("real deeplens smoke test", backend="deeplens")
    result = DeepLensAdapter().simulate_psf_cube(spec, None, tmp_path)

    assert result.status == "succeeded"
    assert result.artifact_refs
    assert (tmp_path / "psf_cube.npz").exists()
    assert result.metrics["backend_capability_level"] in {"smoke", "minimal"}
