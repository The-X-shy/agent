from optiresearch.adapters.deeplens_encoder_strategies import (
    get_deeplens_encoder_strategy,
    list_deeplens_encoder_strategies,
    strategy_to_metadata,
)
from optiresearch.runtime.baselines import ENCODER_TYPES


def test_all_deeplens_encoder_types_have_explicit_strategies():
    strategies = {item.encoder_type: item for item in list_deeplens_encoder_strategies()}

    assert set(strategies) == set(ENCODER_TYPES)
    assert strategies["conventional"].realization_level in {"native", "semi_native", "adapter_proxy"}
    assert strategies["achromatic"].realization_level == "adapter_proxy"
    assert strategies["edof"].realization_level == "adapter_proxy"
    assert strategies["chromatic_coded"].realization_level == "adapter_proxy"
    assert strategies["controlled_chromatic_edof"].realization_level == "adapter_proxy"
    assert all(item.expected_effects for item in strategies.values())


def test_strategy_metadata_marks_proxy_without_claiming_native_support():
    strategy = get_deeplens_encoder_strategy("controlled_chromatic_edof")
    metadata = strategy_to_metadata(strategy)

    assert metadata["encoder_type"] == "controlled_chromatic_edof"
    assert metadata["encoder_behavior_realized"] is True
    assert metadata["encoder_behavior_realization_level"] == "adapter_proxy"
    assert metadata["physical_validation_level"] == "deeplens_base_psf_plus_adapter_proxy"
    assert metadata["proxy_transform_applied"] is True
    assert metadata["proxy_transform_name"]
