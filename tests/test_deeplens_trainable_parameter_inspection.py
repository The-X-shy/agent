from optiresearch.runtime.deeplens_trainable_parameter_inspection import (
    inspect_deeplens_trainable_parameters,
)


def test_inspection_returns_structured_result():
    result = inspect_deeplens_trainable_parameters(device="cpu")
    assert "parameter_count" in result
    assert "evidence_level" in result
