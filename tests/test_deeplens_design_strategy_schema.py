from optiresearch.schemas.deeplens_design_strategy import (
    DeepLensDesignStrategy, DeepLensDesignStrategyResult,
)


def test_strategy_creation():
    s = DeepLensDesignStrategy(
        strategy_id="geolens_curriculum_probe",
        name="GeoLens Curriculum Probe",
        strategy_family="curriculum_learning",
    )
    assert s.strategy_id == "geolens_curriculum_probe"
    assert s.evidence_level == "diagnostic_evidence"


def test_strategy_result_defaults():
    r = DeepLensDesignStrategyResult()
    assert r.status == "dry_run"
    assert r.selected is False
