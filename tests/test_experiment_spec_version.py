import pytest

from optiresearch.schemas.experiment import (
    ExperimentSpec,
    build_default_mock_edof_hsi_experiment,
    validate_experiment_spec_version,
)


def test_default_experiment_spec_uses_v01_schema():
    spec = build_default_mock_edof_hsi_experiment("freeze experiment spec")

    assert spec.schema_version == "0.1"
    assert spec.optical_spec.schema_version == "0.1"
    assert spec.sweep_spec.schema_version == "0.1"
    assert spec.metric_spec.schema_version == "0.1"
    assert validate_experiment_spec_version(spec) is True


def test_experiment_spec_version_validation_rejects_drift():
    payload = build_default_mock_edof_hsi_experiment("detect version drift").model_dump()
    payload["schema_version"] = "0.2"
    drifted = ExperimentSpec(**payload)

    with pytest.raises(ValueError, match="ExperimentSpec schema_version"):
        validate_experiment_spec_version(drifted)
