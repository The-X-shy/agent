from optiresearch.agents.method_builder import MethodBuilder
from optiresearch.schemas.experiment import ExperimentSpec, build_default_mock_edof_hsi_experiment


def test_default_mock_experiment_spec_is_valid():
    spec = build_default_mock_edof_hsi_experiment("Design a mock EDOF-HSI encoder")

    assert isinstance(spec, ExperimentSpec)
    assert spec.backend == "mock_deeplens"
    assert spec.optical_spec.encoder_type == "controlled_chromatic_edof"
    assert spec.optical_spec.sensor_type == "hsi"
    assert spec.sweep_spec.seeds == [42]
    assert spec.metric_spec.primary_metric == "psf_depth_similarity"


def test_method_builder_outputs_experiment_spec():
    spec = MethodBuilder().build_mock_optical_spec("Design a mock EDOF-HSI encoder")

    assert isinstance(spec, ExperimentSpec)
    assert spec.objective == "Design a mock EDOF-HSI encoder"
    assert "spectral_separability" in spec.metric_spec.optical_metrics
