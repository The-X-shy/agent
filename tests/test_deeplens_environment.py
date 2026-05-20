import os

import pytest

from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.cli import main
from optiresearch.schemas.experiment import build_default_mock_edof_hsi_experiment


def test_deeplens_environment_probe_is_structured(capsys):
    environment = DeepLensAdapter().validate_environment()

    assert set(environment).issuperset(
        {
            "available",
            "error_code",
            "message",
            "python_version",
            "deeplens_version",
            "import_path",
            "capabilities",
        }
    )
    assert isinstance(environment["available"], bool)
    assert isinstance(environment["capabilities"], list)
    if not environment["available"]:
        assert environment["error_code"] == "DEEPLENS_NOT_INSTALLED"

    main(["check-deeplens"])
    output = capsys.readouterr().out
    assert "available" in output
    assert "DEEPLENS" in output


@pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_DEEPLENS_TESTS") != "1",
    reason="Real DeepLens smoke tests are opt-in.",
)
def test_real_deeplens_smoke_opt_in(tmp_path):
    experiment = build_default_mock_edof_hsi_experiment("real deeplens opt-in smoke")
    result = DeepLensAdapter().simulate_psf_cube(experiment, None, tmp_path)

    assert result.status in {"succeeded", "failed"}
    if result.status == "succeeded":
        assert result.artifact_refs
