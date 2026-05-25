from optiresearch.optics.deeplens_design_strategy_registry import (
    DeepLensDesignStrategyRegistry, get_deeplens_design_strategy_registry,
)


def test_registry_has_eight_strategies():
    registry = DeepLensDesignStrategyRegistry()
    assert len(registry.list_enabled()) == 8


def test_find_for_diagnosis():
    registry = DeepLensDesignStrategyRegistry()
    strategies = registry.find_for_diagnosis(["unstable_training"])
    ids = {s.strategy_id for s in strategies}
    assert "geolens_curriculum_probe" in ids
    assert "geolens_regularized_probe" in ids


def test_find_by_family():
    registry = DeepLensDesignStrategyRegistry()
    strategies = registry.find_by_family("component_first")
    assert len(strategies) == 2


def test_singleton():
    r1 = get_deeplens_design_strategy_registry()
    r2 = get_deeplens_design_strategy_registry()
    assert r1 is r2
