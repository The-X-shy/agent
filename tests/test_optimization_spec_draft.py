from optiresearch.adapters.deeplens import DeepLensAdapter
from optiresearch.schemas.optimization import OptimizationSpec, build_default_optimization_spec


def test_optimization_spec_draft_defaults():
    spec = build_default_optimization_spec(["psf_depth_similarity"], backend="deeplens")

    assert isinstance(spec, OptimizationSpec)
    assert spec.schema_version == "0.2-draft"
    assert spec.requires_native_support is False
    assert len(spec.optical_variables) == 5


def test_deeplens_optimization_returns_structured_not_available():
    result = DeepLensAdapter(deeplens_module=object()).run_optimization(
        build_default_optimization_spec(["psf_depth_similarity"], backend="deeplens"),
        output_dir=None,
    )

    assert result.status == "failed"
    assert result.errors[0]["code"] == "OPTIMIZATION_NOT_AVAILABLE"
