from optiresearch.adapters.deeplens_encoder_strategies import (
    choose_best_realization_level,
    explain_realization_level,
    get_deeplens_encoder_strategy,
)


def test_choose_best_realization_level_respects_auto_and_forced_modes():
    capabilities = {"paraxial_lens_available": True, "encoder_specific_proxy_available": True}
    api_probe = {"candidate_phase_or_doe_classes": [], "candidate_surface_classes": []}

    assert choose_best_realization_level("conventional", capabilities, api_probe, requested="auto") == "semi_native"
    assert choose_best_realization_level("edof", capabilities, api_probe, requested="auto") == "adapter_proxy"
    assert choose_best_realization_level("edof", capabilities, api_probe, requested="adapter_proxy") == "adapter_proxy"
    assert choose_best_realization_level("edof", capabilities, api_probe, requested="semi_native") == "adapter_proxy"


def test_strategy_contains_phase8_fields():
    strategy = get_deeplens_encoder_strategy("conventional")

    assert "ParaxialLens" in " ".join(strategy.semi_native_plan)
    assert strategy.claim_scope == "baseline DeepLens ParaxialLens behavior"
    assert "semi-native" in explain_realization_level("conventional", "semi_native")
